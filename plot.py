"""
plot.py — generate plots from saved per-algorithm .npz checkpoints.

Conventions:
  - Each experiment lives in `results/{experiment_name}/`.
  - Inside, a `_meta.npz` file describes the experiment (T, n_users, n_items, d, k...).
  - Each algorithm has its own `{algo_name}.npz` with arrays:
      regret           (n_runs, T)
      cpu_time         (n_runs, T)
      memory           (n_runs, T)
      n_user_clusters  (n_runs, T)
      n_arm_groups     (n_runs, T)

This script can be:
  - Imported and called from main.py (after each algo finishes -> incremental refresh).
  - Run standalone:
      python plot.py                     # plots default experiment
      python plot.py <experiment_name>   # plots a specific one
      python plot.py all                 # all experiments under results/

All plots use:
  - Log scale on Y axes (regret, time, memory, clusters).
  - Staggered marker positions per algorithm so curves don't overlap.
  - Fixed colors and markers from algo_registry.STYLES.
"""
import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algo_registry import STYLES, USER_CLUSTERING_ALGOS, ARM_CLUSTERING_ALGOS


def load_results(dir_path):
    """Return (results_dict, meta_dict). results: algo_name -> dict of arrays."""
    results = {}
    meta = {}
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
            meta = {k: data[k].item() if data[k].ndim == 0 else data[k]
                    for k in data.files}
            continue
        algo = fname[:-4]
        algo_data = {k: data[k] for k in data.files}
        # Important: only keep COMPLETED runs. The checkpoint allocates the full
        # (nb_runs, T) array upfront and fills rows as runs finish, so taking
        # the mean over all rows would average in zero-filled rows for runs
        # that haven't run yet, yielding spuriously low values mid-experiment.
        completed = int(algo_data.get('completed_runs', np.array(0)).item()) \
            if 'completed_runs' in algo_data else None
        if completed is not None and completed > 0:
            for key in ('regret', 'cpu_time', 'memory',
                        'n_user_clusters', 'n_arm_groups'):
                if key in algo_data and algo_data[key].ndim == 2:
                    algo_data[key] = algo_data[key][:completed]
        elif completed == 0:
            # Algorithm registered but no run completed yet → skip it from plots.
            continue
        results[algo] = algo_data
    return results, meta


def _markevery_for(i_algo, n_algos, T, n_markers=15):
    """Stagger marker positions for algorithm i_algo out of n_algos."""
    base = max(T // n_markers, 1)
    offset = int((i_algo / max(n_algos, 1)) * base)
    return list(range(offset, T, base))


def plot_experiment(dir_path, log_y=True, save_dir=None):
    """Render two figures (metrics, clustering) for one experiment dir."""
    results, meta = load_results(dir_path)
    if not results:
        print(f"[plot] No results to plot in {dir_path}")
        return

    save_dir = save_dir or dir_path

    T = meta.get('T', list(results.values())[0]['regret'].shape[1])
    if hasattr(T, 'item'):
        T = int(T)
    n_users = meta.get('n_users', '?')
    n_items = meta.get('n_items', '?')
    d = meta.get('d', '?')
    k = meta.get('k', '?')

    # Drop algos whose saved arrays have a different T (stale checkpoints from
    # a previous experiment with different params).
    bad = [name for name, d_ in results.items()
           if d_['regret'].shape[1] != T]
    for name in bad:
        print(f"[plot] skipping stale checkpoint for '{name}' "
              f"(shape {results[name]['regret'].shape}, expected T={T})")
        del results[name]
    if not results:
        print(f"[plot] all results are stale; nothing to plot")
        return
    title_suffix = (f"n_users={n_users}, n_items={n_items}, "
                    f"d={d}, k={k}, T={T}")

    algos = sorted(results.keys())
    n_algos = len(algos)
    time_steps = np.arange(T)

    # ============================================================
    # Figure 1: regret + time + memory
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(22, 5.5))
    for i, name in enumerate(algos):
        d_ = results[name]
        color, marker = STYLES.get(name, ("gray", "o"))
        markevery = _markevery_for(i, n_algos, T)

        # --- regret ---
        m = np.mean(d_['regret'], axis=0)
        s = (np.std(d_['regret'], axis=0)
             if d_['regret'].shape[0] > 1 else None)
        m_safe = np.clip(m, 1e-3, None) if log_y else m
        axes[0].plot(time_steps, m_safe, label=name, color=color,
                     marker=marker, markevery=markevery,
                     markersize=7, linewidth=1.5)
        if s is not None:
            lo = np.clip(m - s, 1e-3, None) if log_y else m - s
            hi = np.clip(m + s, 1e-3, None) if log_y else m + s
            # axes[0].fill_between(time_steps, lo, hi, alpha=0.10, color=color)

        # --- cpu time ---
        m = np.mean(d_['cpu_time'], axis=0)
        m_safe = np.clip(m, 1e-4, None) if log_y else m
        axes[1].plot(time_steps, m_safe, label=name, color=color,
                     marker=marker, markevery=markevery,
                     markersize=7, linewidth=1.5)

        # --- memory ---
        m = np.mean(d_['memory'], axis=0)
        m_safe = np.clip(m, 1e-4, None) if log_y else m
        axes[2].plot(time_steps, m_safe, label=name, color=color,
                     marker=marker, markevery=markevery,
                     markersize=7, linewidth=1.5)

    yscale = "log" if log_y else "linear"
    axes[0].set(xlabel='Round', ylabel='Cumulative Regret', title='Cumulative Regret')
    axes[1].set(xlabel='Round', ylabel='CPU Time (s)', title='CPU Time')
    axes[2].set(xlabel='Round', ylabel='Memory (MB)', title='Memory usage')
    for ax in axes:
        ax.set_yscale(yscale)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=8, loc='best')
    axes[0].set_yscale("linear")
    fig.suptitle(f"[{os.path.basename(dir_path)}] {title_suffix}", fontsize=12)
    plt.tight_layout()
    out1 = os.path.join(save_dir, 'comparison.png')
    plt.savefig(out1, dpi=130, bbox_inches='tight')
    plt.close()

    # ============================================================
    # Figure 2: cluster evolution
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    cluster_algos = [a for a in algos
                     if a in USER_CLUSTERING_ALGOS
                     and 'n_user_clusters' in results[a]]
    for i, name in enumerate(cluster_algos):
        d_ = results[name]
        color, marker = STYLES.get(name, ("gray", "o"))
        markevery = _markevery_for(i, max(len(cluster_algos), 1), T)

        m = np.mean(d_['n_user_clusters'], axis=0)
        m_safe = np.clip(m, 1, None) if log_y else m
        axes[0].plot(time_steps, m_safe, label=name, color=color,
                     marker=marker, markevery=markevery,
                     markersize=7, linewidth=1.5)

        m = np.mean(d_['n_arm_groups'], axis=0)
        m_safe = np.clip(m, 1, None) if log_y else m
        axes[1].plot(time_steps, m_safe, label=name, color=color,
                     marker=marker, markevery=markevery,
                     markersize=7, linewidth=1.5)

    axes[0].set(xlabel='Round', ylabel='# user clusters',
                title='User clusters over time')
    axes[1].set(xlabel='Round',
                ylabel='Total # arm groups (sum over user clusters)',
                title='Arm-group load over time')
    for ax in axes:
        ax.set_yscale(yscale)
        ax.grid(True, which='both', alpha=0.3)
        if cluster_algos:
            ax.legend(fontsize=8, loc='best')
    fig.suptitle(f"[{os.path.basename(dir_path)}] Clustering evolution",
                 fontsize=12)
    plt.tight_layout()
    out2 = os.path.join(save_dir, 'cluster_evolution.png')
    plt.savefig(out2, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"[plot] {dir_path} -> {os.path.basename(out1)}, {os.path.basename(out2)}")

    # NOTE: the "all users mixed" bucket-evolution scatter that used to be
    # generated here has been removed because mixing all users on the same
    # axis is misleading (each LinUCB_IND_* user has its OWN partition, so
    # bucket id 5 for user A and bucket id 5 for user B refer to different
    # arms). Per-user bucket plots are generated by replot.py, which can
    # load the user_stream from _meta.npz or from --user-stream.


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else 'default'
    if target == 'all':
        if not os.path.isdir('results'):
            print("[plot] No results/ directory.")
            return
        for name in sorted(os.listdir('results')):
            path = os.path.join('results', name)
            if os.path.isdir(path):
                plot_experiment(path)
    else:
        plot_experiment(os.path.join('results', target))


if __name__ == '__main__':
    main()