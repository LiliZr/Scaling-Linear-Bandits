"""
replot.py — regenerate publication-quality plots from a saved results directory.

Usage:
    python replot.py path/to/results_dir

What it does:
    * Reads every <algo>.npz in the directory and the matching _meta.npz.
    * Renders two PDF figures inside the same directory:
          comparison.pdf          (regret / CPU time / memory)
          cluster_evolution.pdf   (user clusters / arm groups)
    * Names containing "CA" are displayed as "AB".

Visual encoding (one source of truth, see style_for):
    * Colour encodes the family (LinUCB / LinUCB-IND / SCLUB / CLUB / ...),
      with each family at a different luminance so the curves stay
      distinguishable after grayscale conversion.
    * Marker FILL encodes whether the algorithm is a pure baseline:
          filled marker  -> baseline, no extra technique
          hollow marker  -> at least one technique on top of the baseline
    * Marker SHAPE encodes which technique combination is stacked:
          circle   -> baseline (filled) or RP only (hollow)
          diamond  -> AB only
          square   -> RP + AB
          star     -> KMeans user clustering (full stack)
          x        -> Random sanity baseline

Readability choices for camera-ready figures:
    * x-axis (and the linear regret y-axis) use compact SI labels (200k, 1M)
      with few ticks, so they never overlap.
    * Markers are staggered GLOBALLY (round-robin across families): no two
      curves place a marker at the same iteration, so even fully overlapping
      curves stay traceable by their markers.
    * Curves are drawn in two passes (all lines, then all markers on top) and
      every line carries a thin white halo, so overlapping curves do not bury
      each other. The LinUCB (blue) family is drawn last, on top.
    * The shared legend is grouped: one row per family (base + variants side
      by side), single-variant algorithms collected on the last row.
    * Font sizes are large and exposed at the top of the file.
"""
import argparse
import os
import sys
from collections import OrderedDict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import FuncFormatter, MaxNLocator


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from algo_registry import USER_CLUSTERING_ALGOS  # noqa: E402


# =============================================================================
# Tunable parameters — change here, no need to touch the rest of the file.
# =============================================================================

# --- Sizes (in inches) -------------------------------------------------------
FIG_W_3PANELS = 18.0
FIG_H_3PANELS = 6.0
FIG_W_2PANELS = 13.0
FIG_H_2PANELS = 5.6

# --- Font sizes (large, tuned for a conference paper) ------------------------
FONT_TITLE    = 25
FONT_LABEL    = 23
FONT_TICK     = 20
FONT_LEGEND   = 20
FONT_SUPTITLE = 24
SHOW_SUPTITLE = True   # set False for the camera-ready (caption carries params)

# --- Line / marker properties ------------------------------------------------
LINEWIDTH_DEFAULT  = 2.4
LINEWIDTH_KMEANS   = 2.8
LINEWIDTH_RANDOM   = 2.0
MARKERSIZE         = 9
MARKEREDGEWIDTH    = 1.7
LINE_ALPHA         = 0.65   # <1 so overlapping curves blend instead of hiding
N_MARKERS_PER_LINE = 5      # fewer markers => more spacing along each curve

# --- Regret inset (zoom on the low cluster, where blue/green nearly overlap) -
SHOW_REGRET_INSET = False    # default off; enable per call (--zoom / show_regret_inset=True)
INSET_X_START     = 0.5      # inset shows iterations from this fraction of T to T
INSET_KEEP_PCTL   = 65       # zoom on curves whose final regret <= this percentile
INSET_RECT        = (0.07, 0.50, 0.47, 0.46)   # x, y, w, h in axes fraction

# --- Axis scale defaults -----------------------------------------------------
LOG_Y_REGRET = False
LOG_Y_CPU    = True
LOG_Y_MEM    = True
LOG_Y_CLUSTR = True

# --- Ticks -------------------------------------------------------------------
N_XTICKS = 5
N_YTICKS_LINEAR = 6

# --- Output ------------------------------------------------------------------
DPI = 200
SAVE_FORMAT = "pdf"


# =============================================================================
# Colours and family ordering.
# =============================================================================

FAMILY_COLOR = {
    "Random":     "#9e9e9e",   # mid grey   (sanity baseline)
    "LinUCB":     "#1f77b4",   # blue
    "LinUCB_IND": "#d62728",   # red
    "SCLUB":      "#2ca02c",   # green
    "CLUB":       "#9467bd",   # purple
    "Bandit":     "#000000",   # black
}

# Order in which families appear in the legend (lower = first).
FAMILY_PRIORITY = {
    "LinUCB": 0, "LinUCB_IND": 1, "SCLUB": 2,
    "CLUB": 3, "Bandit": 4, "Random": 5,
}

# Marker shape per technique combination.
MARKER_BASELINE = "o"   # filled circle  : no technique
MARKER_RP       = "o"   # hollow circle  : RP only
MARKER_AB       = "D"   # hollow diamond : AB only
MARKER_RP_AB    = "s"   # hollow square  : RP + AB
MARKER_KMEANS   = "*"   # hollow star    : full stack (with KMeans)

# Canonical column order for the legend, by technique signature
# (has_rp, has_ab, has_km). Every family puts each variant in the SAME column,
# leaving a gap where it lacks the variant, so columns line up across rows:
#   col0 baseline | col1 RP | col2 RP+AB | col3 RP+AB+KMeans | col4 AB | ...
VARIANT_ORDER = [
    (False, False, False),   # baseline
    (True,  False, False),   # RP
    (True,  True,  False),   # RP + AB
    (True,  True,  True),    # RP + AB + KMeans
    (False, True,  False),   # AB only
    (False, True,  True),    # AB + KMeans
    (True,  False, True),    # RP + KMeans
    (False, False, True),    # KMeans only
]


def _family_of(name: str) -> str:
    if name == "Random":
        return "Random"
    if name.startswith("Bandit"):
        return "Bandit"
    if name.startswith("LinUCB_IND"):
        return "LinUCB_IND"
    if name.startswith("LinUCB"):
        return "LinUCB"
    if name.startswith("SCLUB"):
        return "SCLUB"
    if name.startswith("CLUB"):
        return "CLUB"
    return "LinUCB"


def _has_rp(name: str) -> bool:
    return "_RP" in name or name.endswith("_RP")


def _has_ab(name: str) -> bool:
    """AB = the old CA name (arm clustering / arm bucketing)."""
    return "_CA" in name


def _has_kmeans(name: str) -> bool:
    return "KMeans" in name


def _variant_col(name: str) -> int:
    """Rank of `name` in the canonical variant order (see VARIANT_ORDER).

    Used both to order members inside a family and to keep the legend columns
    consistent (baseline, RP, RP+AB, ... in the same order for every family).
    """
    sig = (_has_rp(name), _has_ab(name), _has_kmeans(name))
    try:
        return VARIANT_ORDER.index(sig)
    except ValueError:
        return len(VARIANT_ORDER)


def style_for(name: str) -> dict:
    """Return all the matplotlib kwargs needed to plot `name` consistently.

    Fill = baseline vs technique; shape = which technique combination.
    """
    fam = _family_of(name)
    color = FAMILY_COLOR[fam]

    if name == "Random":
        return dict(
            color=color, marker="x", markerfacecolor=color,
            markeredgecolor=color, markeredgewidth=MARKEREDGEWIDTH,
            linestyle=(0, (3, 2)), linewidth=LINEWIDTH_RANDOM,
            markersize=MARKERSIZE,
        )

    has_rp, has_ab, has_km = _has_rp(name), _has_ab(name), _has_kmeans(name)

    if has_km:
        marker, linestyle, lw = MARKER_KMEANS, (0, (4, 2)), LINEWIDTH_KMEANS
        msize = MARKERSIZE * 1.3
    elif has_rp and has_ab:
        marker, linestyle, lw, msize = MARKER_RP_AB, "-", LINEWIDTH_DEFAULT, MARKERSIZE
    elif has_rp:
        marker, linestyle, lw, msize = MARKER_RP, "-", LINEWIDTH_DEFAULT, MARKERSIZE
    elif has_ab:
        marker, linestyle, lw, msize = MARKER_AB, "-", LINEWIDTH_DEFAULT, MARKERSIZE
    else:
        marker, linestyle, lw, msize = MARKER_BASELINE, "-", LINEWIDTH_DEFAULT, MARKERSIZE

    is_baseline = not (has_rp or has_ab or has_km)
    markerfacecolor = color if is_baseline else "white"

    return dict(
        color=color, marker=marker,
        markerfacecolor=markerfacecolor,
        markeredgecolor=color,
        markeredgewidth=MARKEREDGEWIDTH,
        linestyle=linestyle, linewidth=lw,
        markersize=msize,
    )


def display_name(name: str) -> str:
    """Rewrite the algorithm name for display: CA -> AB."""
    return name.replace("_CA", "_AB")


# =============================================================================
# Ordering: group algorithms by family for the legend, interleave for markers.
# =============================================================================

def group_for_display(algos):
    """Return (flat_order, legend_rows).

    flat_order  : algos family by family, used as the plotting order.
    legend_rows : one row per multi-variant family, members sorted by the
                  canonical variant order (baseline, RP, RP+AB, then the
                  family-specific extra) and left-packed, so columns 0..2 read
                  baseline / RP / RP+AB in every row. Single-variant algorithms
                  (Random, CLUB, ...) are collected on a final row.
    """
    fam_members = OrderedDict()
    for name in algos:
        fam_members.setdefault(_family_of(name), []).append(name)
    for fam in fam_members:
        fam_members[fam].sort(key=_variant_col)

    multi = [f for f in fam_members if len(fam_members[f]) > 1]
    single = [f for f in fam_members if len(fam_members[f]) == 1]
    multi.sort(key=lambda f: FAMILY_PRIORITY.get(f, 99))
    single.sort(key=lambda f: FAMILY_PRIORITY.get(f, 99))

    rows = [list(fam_members[f]) for f in multi]
    singletons = [fam_members[f][0] for f in single]
    if singletons:
        rows.append(singletons)

    flat = [name for row in rows for name in row]
    return flat, rows


def _marker_layout(algos, T, n_markers=N_MARKERS_PER_LINE):
    """Give each curve a marker phase, separated by family.

    Each family owns a contiguous sub-band of the inter-marker gap (family 0 in
    the first third, family 1 in the second, ...), and its members are spread
    inside that band. Consequence: markers of two different families never land
    at the same iteration, so close curves of different colour (e.g. blue vs
    green) stay clearly apart; same-family members are spread within the band.
    """
    base = max(T // n_markers, 1)
    fam_members = OrderedDict()
    for name in algos:
        fam_members.setdefault(_family_of(name), []).append(name)
    for fam in fam_members:
        fam_members[fam].sort(key=_variant_col)

    fams = list(fam_members.items())
    n_fams = max(len(fams), 1)
    offsets = {}
    for fi, (_fam, members) in enumerate(fams):
        band_lo = (fi / n_fams) * base
        span = base / n_fams
        m = len(members)
        for rank, name in enumerate(members):
            frac = (rank + 0.5) / m            # centre members inside the band
            offsets[name] = int(band_lo + frac * span)
    return base, offsets


# =============================================================================
# Tick formatting
# =============================================================================

def _si_fmt(x, pos=None):
    """Compact SI tick label: 0, 200k, 1M ..."""
    if x == 0:
        return "0"
    ax = abs(x)
    if ax >= 1e6:
        return f"{x / 1e6:g}M"
    if ax >= 1e3:
        return f"{x / 1e3:g}k"
    return f"{x:g}"


def _style_x_axis(ax):
    ax.xaxis.set_major_locator(MaxNLocator(nbins=N_XTICKS))
    ax.xaxis.set_major_formatter(FuncFormatter(_si_fmt))


def _style_y_linear(ax):
    ax.yaxis.set_major_locator(MaxNLocator(nbins=N_YTICKS_LINEAR))
    ax.yaxis.set_major_formatter(FuncFormatter(_si_fmt))


# =============================================================================
# IO
# =============================================================================

def load_results(dir_path):
    results, meta = {}, {}
    if not os.path.isdir(dir_path):
        return results, meta
    for fname in sorted(os.listdir(dir_path)):
        if not fname.endswith(".npz"):
            continue
        path = os.path.join(dir_path, fname)
        try:
            data = np.load(path, allow_pickle=True)
        except Exception:
            continue
        if fname == "_meta.npz":
            meta = {k: (data[k].item() if data[k].ndim == 0 else data[k])
                    for k in data.files}
            continue
        algo = fname[:-4]
        algo_data = {k: data[k] for k in data.files}
        completed = (int(algo_data['completed_runs'].item())
                     if 'completed_runs' in algo_data else None)
        if completed == 0:
            continue
        if completed is not None and completed > 0:
            for key in ('regret', 'cpu_time', 'memory',
                        'n_user_clusters', 'n_arm_groups'):
                if key in algo_data and algo_data[key].ndim == 2:
                    algo_data[key] = algo_data[key][:completed]
        results[algo] = algo_data
    return results, meta


# =============================================================================
# Drawing primitives (two passes: lines first, markers on top)
# =============================================================================

def _line_kwargs(style):
    return dict(
        color=style["color"], linestyle=style["linestyle"],
        linewidth=style["linewidth"], solid_capstyle="round",
    )


def _draw_line(ax, x, y, style, log_y, eps, zorder=2):
    y_safe = np.clip(y, eps, None) if log_y else y
    ax.plot(
        x, y_safe, marker="none", zorder=zorder, alpha=LINE_ALPHA,
        **_line_kwargs(style),
    )
    return y_safe


def _draw_markers(ax, x, y_safe, style, markevery, zorder=4):
    idx = np.asarray(markevery, dtype=int)
    idx = idx[idx < len(x)]
    ax.plot(
        x[idx], y_safe[idx], linestyle="none",
        color=style["color"], marker=style["marker"],
        markerfacecolor=style["markerfacecolor"],
        markeredgecolor=style["markeredgecolor"],
        markeredgewidth=style["markeredgewidth"],
        markersize=style["markersize"], zorder=zorder,
    )


def _plot_panel(ax, x, curves, key, log_y, eps):
    """curves: list of (name, style, markevery, data_dict).

    Lines (semi-transparent) are drawn first, then all markers (opaque) on top,
    so overlapping curves blend rather than hide each other and each curve stays
    identifiable by its staggered markers.
    """
    y_safe = {}
    for name, style, _, d_ in curves:
        y_safe[name] = _draw_line(ax, x, d_[key].mean(axis=0), style, log_y, eps)
    for name, style, me, _ in curves:
        _draw_markers(ax, x, y_safe[name], style, me)


def _add_regret_inset(ax, x, curves):
    """Zoom on the low-regret cluster, where blue/green curves nearly overlap.

    The x window is the tail (>= INSET_X_START * T); the y window is fitted to
    the curves whose final regret is in the lowest INSET_KEEP_PCTL percent.
    """
    T = len(x)
    finals = {name: d_['regret'].mean(axis=0)[-1] for name, _, _, d_ in curves}
    thr = np.percentile(list(finals.values()), INSET_KEEP_PCTL)
    keep = [c for c in curves if finals[c[0]] <= thr]
    if len(keep) < 2:
        return

    x0 = int(T * INSET_X_START)
    seg = [d_['regret'].mean(axis=0)[x0:] for _, _, _, d_ in keep]
    ymin = min(s.min() for s in seg)
    ymax = max(s.max() for s in seg)
    margin = 0.06 * (ymax - ymin + 1e-9)

    axin = ax.inset_axes(INSET_RECT)
    _plot_panel(axin, x, keep, 'regret', LOG_Y_REGRET, 1e-6)
    axin.set_xlim(x0, T - 1)
    axin.set_ylim(ymin - margin, ymax + margin)
    axin.xaxis.set_major_locator(MaxNLocator(nbins=3))
    axin.yaxis.set_major_locator(MaxNLocator(nbins=4))
    axin.xaxis.set_major_formatter(FuncFormatter(_si_fmt))
    axin.yaxis.set_major_formatter(FuncFormatter(_si_fmt))
    axin.tick_params(axis='both', labelsize=FONT_TICK * 0.62)
    axin.grid(True, which='major', alpha=0.30)
    ax.indicate_inset_zoom(axin, edgecolor="0.35", alpha=0.85, linewidth=1.4)


def _grouped_legend(fig, rows, handle_by_name):
    """Shared legend with one row per family (matplotlib fills column-major)."""
    nrows = len(rows)
    ncol = max(len(r) for r in rows)
    grid = [list(r) + [None] * (ncol - len(r)) for r in rows]

    handles, labels = [], []
    for c in range(ncol):
        for r in range(nrows):
            name = grid[r][c]
            if name is None:
                handles.append(mlines.Line2D([], [], linestyle="none",
                                              marker="none"))
                labels.append("")
            else:
                handles.append(handle_by_name[name])
                labels.append(display_name(name))

    fig.legend(
        handles=handles, labels=labels,
        loc="upper center", bbox_to_anchor=(0.5, 0.04),
        ncol=ncol, fontsize=FONT_LEGEND,
        frameon=True, framealpha=0.95, edgecolor="0.8",
        borderaxespad=0.4, handletextpad=0.5,
        columnspacing=1.5, labelspacing=0.7,
    )


def _legend_handles(names):
    return {name: mlines.Line2D([], [], **style_for(name)) for name in names}


# =============================================================================
# Plotting
# =============================================================================

def _load_user_stream(dir_path, user_stream_path, meta, T):
    """Try to load the user stream of integers used during the experiment.

    Resolution order:
      1. Explicit user_stream_path (.npy or .npz with key 'user_stream').
      2. meta['user_stream'] if the experiment was launched with the version
         of main.py that saves the stream in _meta.npz.
      3. None if nothing works (the per-user plot is silently skipped).

    Returns a 1-D int32 array of length T, or None.
    """
    candidates = []
    if user_stream_path is not None:
        candidates.append(user_stream_path)
    if isinstance(meta, dict) and 'user_stream' in meta:
        arr = np.asarray(meta['user_stream']).ravel()
        if arr.size >= T:
            return arr[:T].astype(np.int32)

    for cand in candidates:
        try:
            if cand.endswith('.npz'):
                d = np.load(cand)
                if 'user_stream' in d.files:
                    arr = np.asarray(d['user_stream']).ravel()
                else:
                    arr = np.asarray(d[d.files[0]]).ravel()
            else:
                arr = np.asarray(np.load(cand)).ravel()
            if arr.size >= T:
                return arr[:T].astype(np.int32)
            print(f"[replot] user_stream too short "
                  f"({arr.size} < T={T}), ignoring.")
        except Exception as e:
            print(f"[replot] could not load user stream from {cand}: {e}")
    return None


def plot_experiment(dir_path, save_dir=None, show_regret_inset=None,
                    user_stream_path=None, top_users=1):
    """Render the comparison and cluster-evolution figures.

    show_regret_inset: None -> use the SHOW_REGRET_INSET default; True/False
    overrides it for this call.
    """
    inset = SHOW_REGRET_INSET if show_regret_inset is None else show_regret_inset

    results, meta = load_results(dir_path)
    if not results:
        print(f"[replot] No results to plot in {dir_path}")
        return

    save_dir = save_dir or dir_path

    T = meta.get('T', list(results.values())[0]['regret'].shape[1])
    if hasattr(T, 'item'):
        T = int(T)
    n_users = meta.get('n_users', '?')
    n_items = meta.get('n_items', '?')
    d = meta.get('d', '?')
    k = meta.get('k', '?')

    bad = [n for n, dd in results.items() if dd['regret'].shape[1] != T]
    for name in bad:
        print(f"[replot] skipping stale checkpoint for '{name}'")
        del results[name]
    if not results:
        return

    title_suffix = (f"n_users={n_users}, n_items={n_items}, "
                    f"d={d}, k={k}, T={T}")
    flat_algos, legend_rows = group_for_display(list(results.keys()))
    base, offsets = _marker_layout(flat_algos, T)
    iterations = np.arange(T)

    curves = [(name, style_for(name), list(range(offsets[name], T, base)),
               results[name]) for name in flat_algos]

    # ===========================================================
    # Figure 1: regret + CPU + memory (shared grouped legend below)
    # ===========================================================
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W_3PANELS, FIG_H_3PANELS))

    _plot_panel(axes[0], iterations, curves, 'regret', LOG_Y_REGRET, 1e-6)
    _plot_panel(axes[1], iterations, curves, 'cpu_time', LOG_Y_CPU, 1e-4)
    _plot_panel(axes[2], iterations, curves, 'memory', LOG_Y_MEM, 1e-4)

    if inset and not LOG_Y_REGRET:
        _add_regret_inset(axes[0], iterations, curves)

    titles = ("Cumulative regret", "CPU time", "Memory")
    ylabels = ("Cumulative regret", "CPU time (s)", "Memory (MB)")
    log_flags = (LOG_Y_REGRET, LOG_Y_CPU, LOG_Y_MEM)
    for ax, title, ylab, logy in zip(axes, titles, ylabels, log_flags):
        ax.set_title(title, fontsize=FONT_TITLE, pad=10)
        ax.set_xlabel("Iteration", fontsize=FONT_LABEL)
        ax.set_ylabel(ylab, fontsize=FONT_LABEL)
        ax.tick_params(axis='both', which='major', labelsize=FONT_TICK)
        ax.grid(True, which='major', alpha=0.30)
        ax.grid(True, which='minor', alpha=0.12)
        _style_x_axis(ax)
        if logy:
            ax.set_yscale('log')
        else:
            _style_y_linear(ax)

    if SHOW_SUPTITLE:
        fig.suptitle(f"[{os.path.basename(dir_path)}]  {title_suffix}",
                     fontsize=FONT_SUPTITLE, y=1.02)

    _grouped_legend(fig, legend_rows, _legend_handles(flat_algos))

    top = 0.90 if SHOW_SUPTITLE else 0.95
    fig.subplots_adjust(left=0.06, right=0.99, top=top, bottom=0.16, wspace=0.24)
    out1 = os.path.join(save_dir, f"comparison.{SAVE_FORMAT}")
    fig.savefig(out1, dpi=DPI, bbox_inches='tight', format=SAVE_FORMAT)
    plt.close(fig)

    # ===========================================================
    # Figure 2: cluster evolution (only for algos that cluster users)
    # ===========================================================
    cluster_pool = [a for a in flat_algos
                    if a in USER_CLUSTERING_ALGOS
                    and 'n_user_clusters' in results[a]]
    out2 = None
    if cluster_pool:
        _, cl_legend_rows = group_for_display(cluster_pool)
        cl_base, cl_offsets = _marker_layout(cluster_pool, T)
        cl_curves = [(name, style_for(name),
                      list(range(cl_offsets[name], T, cl_base)), results[name])
                     for name in cluster_pool]

        fig, axes = plt.subplots(1, 2, figsize=(FIG_W_2PANELS, FIG_H_2PANELS))
        _plot_panel(axes[0], iterations, cl_curves, 'n_user_clusters',
                    LOG_Y_CLUSTR, 1)
        _plot_panel(axes[1], iterations, cl_curves, 'n_arm_groups',
                    LOG_Y_CLUSTR, 1)

        axes[0].set_title("User clusters over time", fontsize=FONT_TITLE, pad=10)
        axes[0].set_ylabel("# user clusters", fontsize=FONT_LABEL)
        axes[1].set_title("Arm groups over time", fontsize=FONT_TITLE, pad=10)
        axes[1].set_ylabel("Total # arm groups", fontsize=FONT_LABEL)
        for ax in axes:
            ax.set_xlabel("Iteration", fontsize=FONT_LABEL)
            ax.tick_params(axis='both', which='major', labelsize=FONT_TICK)
            ax.grid(True, which='major', alpha=0.30)
            ax.grid(True, which='minor', alpha=0.12)
            _style_x_axis(ax)
            if LOG_Y_CLUSTR:
                ax.set_yscale('log')
            else:
                _style_y_linear(ax)

        if SHOW_SUPTITLE:
            fig.suptitle(f"[{os.path.basename(dir_path)}]  Clustering evolution",
                         fontsize=FONT_SUPTITLE, y=1.02)
        _grouped_legend(fig, cl_legend_rows, _legend_handles(cluster_pool))

        top = 0.90 if SHOW_SUPTITLE else 0.95
        fig.subplots_adjust(left=0.08, right=0.99, top=top, bottom=0.18,
                            wspace=0.24)
        out2 = os.path.join(save_dir, f"cluster_evolution.{SAVE_FORMAT}")
        fig.savefig(out2, dpi=DPI, bbox_inches='tight', format=SAVE_FORMAT)
        plt.close(fig)

    # ===========================================================
    # Bucket-per-user line plot (replaces the old "all users mixed"
    # bucket_evolution scatter which was hard to read).
    #
    # X axis = "k-th observation of this user" (1, 2, 3, ...). This
    # aligns the start of every user's trajectory at k=1, regardless of
    # when the user first appeared in the stream, which lets the eye
    # compare exploration trajectories directly.
    # Y axis = bucket id selected at that observation.
    # Each top user has its own colour (palette of strongly distinguishable
    # colours, cycling if more than 6). Lines connect successive
    # observations of the same user; gaps in the global stream are
    # invisible because we collapse them by definition.
    #
    # Only meaningful for LinUCB_IND_* family algos (where each user has
    # its own arm partition).
    # ===========================================================

    # First detect which algos have bucket tracking enabled at all.
    bucket_algos = []
    for name in flat_algos:
        d_ = results[name]
        if 'bucket_over_time' not in d_:
            continue
        bot = d_['bucket_over_time']
        if bot.ndim != 2 or bot.shape[1] != T:
            continue
        valid_rows = [r for r in range(bot.shape[0]) if (bot[r] >= 0).any()]
        if valid_rows:
            bucket_algos.append((name, bot, valid_rows))

    out3 = None  # kept as None — the old "all users mixed" scatter is removed.
    out4 = None
    user_stream = _load_user_stream(dir_path, user_stream_path, meta, T)
    # Hand-picked palette of strongly distinguishable colours that also
    # survive grayscale conversion (varied luminance). Cycled if needed.
    USER_PALETTE = [
        '#d62728',  # red
        '#1f77b4',  # blue
        '#2ca02c',  # green
        '#ff7f0e',  # orange
        '#9467bd',  # purple
        '#000000',  # black
    ]
    if user_stream is not None and bucket_algos:
        per_user_algos = [(name, bot, vr) for (name, bot, vr) in bucket_algos
                          if 'LinUCB_IND' in name and '_CA' in name]
        if per_user_algos:
            # Determine the top-N most active users.
            uniq, counts = np.unique(user_stream, return_counts=True)
            order = np.argsort(-counts)
            top_uids = uniq[order[:max(1, top_users)]]
            top_counts = counts[order[:max(1, top_users)]]
            # Max horizon on the X axis = the most observed user's count.
            x_max = int(top_counts[0])

            n = len(per_user_algos)
            fig_w = max(FIG_W_3PANELS, 5.5 * n)
            fig_h = FIG_H_3PANELS
            fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h),
                                     squeeze=False)
            axes = axes[0]
            for ax, (name, bot, valid_rows) in zip(axes, per_user_algos):
                # Use the first valid run.
                r = valid_rows[0]
                bucket_row = bot[r]
                for ui, (uid, cnt) in enumerate(zip(top_uids, top_counts)):
                    # Indices in the GLOBAL stream where this user acted.
                    mask = (user_stream == uid) & (bucket_row >= 0)
                    global_ts = np.where(mask)[0]
                    if len(global_ts) == 0:
                        continue
                    # The X coordinate is now 1, 2, 3, ... = the user's own
                    # observation index, NOT the global iteration.
                    ys = bucket_row[global_ts]
                    xs = np.arange(1, len(ys) + 1)
                    # Subsample if a user has too many observations (keeps
                    # the PDF light on large datasets).
                    if len(xs) > 4000:
                        step = len(xs) // 2000
                        xs = xs[::step]
                        ys = ys[::step]
                    color = USER_PALETTE[ui % len(USER_PALETTE)]
                    # Marker size and line width: small markers + thin
                    # lines so several users overlay without saturating.
                    ms = max(3, 6 - top_users)
                    lw = max(0.6, 1.4 - 0.15 * top_users)
                    alpha = 0.9 if top_users == 1 else 0.6
                    ax.plot(xs, ys, color=color, marker='o',
                            markersize=ms, markeredgewidth=0,
                            linewidth=lw, alpha=alpha,
                            label=f"user {int(uid)} ({int(cnt)} obs)")
                ax.set_title(display_name(name), fontsize=FONT_TITLE)
                ax.set_xlabel("k-th observation of the user",
                              fontsize=FONT_LABEL)
                ax.set_ylabel("Bucket id (selected)", fontsize=FONT_LABEL)
                ax.set_xlim(0.5, x_max + 0.5)
                ax.tick_params(axis='both', which='major',
                               labelsize=FONT_TICK)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=max(FONT_LEGEND - 2, 9),
                          loc='best', markerscale=1.0,
                          framealpha=0.85)
            if SHOW_SUPTITLE:
                fig.suptitle(
                    f"[{os.path.basename(dir_path)}]  "
                    f"Per-user bucket selection "
                    f"(top {len(top_uids)} user{'s' if len(top_uids)>1 else ''}, "
                    f"x = own observation index)",
                    fontsize=FONT_SUPTITLE, y=1.02,
                )
            fig.subplots_adjust(left=0.05, right=0.99,
                                top=0.90 if SHOW_SUPTITLE else 0.95,
                                bottom=0.16, wspace=0.22)
            out4 = os.path.join(save_dir,
                                f"bucket_evolution_per_user.{SAVE_FORMAT}")
            fig.savefig(out4, dpi=DPI, bbox_inches='tight',
                        format=SAVE_FORMAT)
            plt.close(fig)

    print(f"[replot] saved {out1}"
          + (f", {out2}" if out2 else "")
          + (f", {out3}" if out3 else "")
          + (f", {out4}" if out4 else ""))


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Regenerate publication-quality PDF plots from a results "
                    "directory. Algorithms with 'CA' in their name are "
                    "displayed as 'AB'.",
    )
    parser.add_argument(
        "results_dir",
        help="Path to the results directory (e.g. results/medium_steam).",
    )
    parser.add_argument(
        "--zoom", action="store_true",
        help="Add a zoom inset on the low-regret cluster (off by default).",
    )
    parser.add_argument(
        "--user-stream", default=None,
        help="Optional path to a .npy or .npz file containing the user "
             "stream of integers used during the experiment. If omitted, "
             "the script looks for `user_stream` inside _meta.npz. The "
             "per-user bucket scatter only renders when a stream is "
             "available.",
    )
    parser.add_argument(
        "--top-users", type=int, default=1,
        help="Number of top-activity users to highlight in the per-user "
             "bucket plot (default 1, the single most active user). "
             "Max 6 (limited by the distinguishable color palette).",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"ERROR: '{args.results_dir}' is not a directory.",
              file=sys.stderr)
        sys.exit(1)

    plot_experiment(args.results_dir,
                    show_regret_inset=args.zoom,
                    user_stream_path=args.user_stream,
                    top_users=args.top_users)


if __name__ == "__main__":
    main()