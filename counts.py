"""
recluster_histogram.py — reconstruct and visualise per-user recluster counts
from a saved experiment, without re-running any algorithm.

USAGE
=====
    python recluster_histogram.py results/<experiment_dir> \
        --algo LinUCB_IND_RP_CA \
        --arm-recluster-freq 500

    # Several algos at once (one figure per algo, side-by-side):
    python recluster_histogram.py results/<exp_dir> \
        --algo LinUCB_IND_RP_CA LinUCB_IND_RP_CA_KMeans \
        --arm-recluster-freq 500

WHAT IT DOES
============
For algorithms in the LinUCB_IND_*_CA family, an arm-clustering recluster
fires deterministically when T_u[user] % arm_recluster_freq == 0 (plus one
init recluster per user). Therefore the *number of times* each user has
been reclusterised across a run is fully determined by:

  - the user stream (saved in _meta.npz['user_stream'])
  - the arm_recluster_freq value used

The script:
  1. Loads user_stream from _meta.npz (or --user-stream PATH if given).
  2. Counts T_u for every user.
  3. Computes per-user recluster_count = floor(T_u / freq) + 1 init.
  4. Plots:
     - Histogram of per-user recluster counts
     - Bar chart of "users observed N times" (T_u distribution)
     - Cumulative number of reclusters over time

The same reconstruction works for SCLUB_*_CA but in those algos the
counter is per *user-cluster*, not per user. Since cluster assignments
change over time and are not currently saved per-round, this script only
handles the per-user case here.

OUTPUT
======
A PDF named recluster_histogram.<algo>.pdf in the experiment directory,
plus a printed summary to stdout.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_stream(exp_dir, user_stream_path):
    """Load the user stream from _meta.npz or from an explicit path."""
    if user_stream_path is not None:
        if user_stream_path.endswith('.npz'):
            d = np.load(user_stream_path)
            if 'user_stream' in d.files:
                return np.asarray(d['user_stream']).ravel()
            return np.asarray(d[d.files[0]]).ravel()
        return np.asarray(np.load(user_stream_path)).ravel()
    meta_path = os.path.join(exp_dir, '_meta.npz')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"No _meta.npz in {exp_dir} and no --user-stream provided. "
            f"Either re-run the experiment with the version of main.py "
            f"that saves user_stream, or pass --user-stream PATH."
        )
    m = np.load(meta_path)
    if 'user_stream' not in m.files:
        raise KeyError(
            f"_meta.npz exists but has no 'user_stream' field. "
            f"This run is from an older version of main.py. "
            f"Use --user-stream PATH pointing to the dataset npz, "
            f"or to the original raw stream."
        )
    return np.asarray(m['user_stream']).ravel()


def _per_user_recluster_counts(user_stream, freq, T=None):
    """For each user observed in the stream, how many recluster fires?

    Returns a dict user_id -> count.
    count = number of inits (always 1) + post-init reclusters (T_u // freq).
    """
    if T is not None and T < len(user_stream):
        user_stream = user_stream[:T]
    uids, T_u = np.unique(user_stream, return_counts=True)
    # Init is triggered once per user when they first appear (cost is small
    # because theta=0). Post-init reclusters happen every `freq` user
    # observations.
    post_init = T_u // freq
    total = 1 + post_init  # 1 init + post-init reclusters
    return {int(u): {'T_u': int(t), 'post_init': int(p), 'total': int(c)}
            for u, t, p, c in zip(uids, T_u, post_init, total)}


def _cumulative_reclusters_over_time(user_stream, freq):
    """Compute the cumulative number of recluster fires as a function of t.

    Returns a 1D array of length len(user_stream) such that
    cumulative[t] = total number of reclusters fired up to and including
    round t (init + post-init).

    Includes both kinds of events:
      - init: fires the first time a user is seen
      - post-init: fires when the user's running T_u becomes a multiple of freq
    """
    T = len(user_stream)
    cum = np.zeros(T, dtype=np.int64)
    counts_so_far = {}
    seen = set()
    for t in range(T):
        u = int(user_stream[t])
        if u not in seen:
            seen.add(u)
            counts_so_far[u] = 1   # the algo runs an init recluster
            ev = 1                  # this round triggers an init recluster
        else:
            counts_so_far[u] += 1
            # post-init recluster fires when T_u becomes a multiple of freq
            ev = 1 if counts_so_far[u] % freq == 0 else 0
        cum[t] = (cum[t-1] if t > 0 else 0) + ev
    return cum


def plot_recluster_histogram(exp_dir, algos, freq, user_stream_path=None,
                              show_init=True):
    """Build the figures and write the PDF.

    One figure per algo (since each algo can in principle have a different
    arm_recluster_freq, although the CLI here passes one common value).
    """
    user_stream = _load_stream(exp_dir, user_stream_path)
    T = len(user_stream)
    print(f"\n[recluster_histogram] {os.path.basename(exp_dir)}")
    print(f"  T = {T}, distinct users = {len(np.unique(user_stream))}")
    print(f"  arm_recluster_freq = {freq}\n")

    for algo in algos:
        # Stats per user.
        per_user = _per_user_recluster_counts(user_stream, freq, T=T)
        counts = np.array([d['total'] for d in per_user.values()])
        post_init_counts = np.array(
            [d['post_init'] for d in per_user.values()])
        T_u_vals = np.array([d['T_u'] for d in per_user.values()])

        n_users = len(per_user)
        total_reclusters = int(counts.sum())
        total_post_init = int(post_init_counts.sum())
        n_user_with_post = int((post_init_counts > 0).sum())

        print(f"  --- {algo} ---")
        print(f"    users seen                 = {n_users}")
        print(f"    total recluster events     = {total_reclusters}")
        print(f"      init events              = {n_users}")
        print(f"      post-init events         = {total_post_init}")
        print(f"    users with >=1 post-init   = {n_user_with_post} "
              f"({100*n_user_with_post/max(1,n_users):.1f}%)")
        if total_post_init == 0:
            print(f"    ==> NO POST-INIT RECLUSTER FIRED. "
                  f"T_u never reaches {freq} for any user.")
            print(f"        max T_u = {int(T_u_vals.max())}, "
                  f"mean = {T_u_vals.mean():.1f}, "
                  f"median = {int(np.median(T_u_vals))}")
        else:
            print(f"    max post-init per user     = {int(post_init_counts.max())}")
            print(f"    mean T_u                   = {T_u_vals.mean():.1f}")
            print(f"    max T_u                    = {int(T_u_vals.max())}")
        print()

        # Build the figure.
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

        # --- Panel 1: histogram of post-init recluster counts per user ---
        ax = axes[0]
        if total_post_init > 0:
            max_count = int(post_init_counts.max())
            bins = np.arange(-0.5, max_count + 1.5, 1)
            ax.hist(post_init_counts, bins=bins, color='#1f77b4',
                    edgecolor='black', linewidth=0.5)
            ax.set_xticks(range(0, max_count + 1))
        else:
            ax.hist([0] * n_users, bins=[-0.5, 0.5, 1.5],
                    color='#1f77b4', edgecolor='black', linewidth=0.5)
            ax.set_xticks([0])
            ax.text(0.5, 0.7, "No user reached the recluster threshold",
                    transform=ax.transAxes, ha='center',
                    fontsize=11, color='#d62728', fontweight='bold')
        ax.set_xlabel("Number of post-init reclusters per user",
                      fontsize=13)
        ax.set_ylabel("Number of users", fontsize=13)
        ax.set_title(f"{algo} — post-init recluster events per user",
                     fontsize=13)
        ax.grid(True, axis='y', alpha=0.3)
        ax.tick_params(axis='both', labelsize=11)

        # --- Panel 2: histogram of T_u (user activity distribution) ---
        ax = axes[1]
        # Buckets adapted to the distribution.
        max_tu = int(T_u_vals.max())
        if max_tu <= 30:
            bins = np.arange(0.5, max_tu + 1.5, 1)
        else:
            bins = 30
        ax.hist(T_u_vals, bins=bins, color='#2ca02c',
                edgecolor='black', linewidth=0.5)
        ax.axvline(x=freq, color='#d62728', linestyle='--', linewidth=1.5,
                   label=f"arm_recluster_freq = {freq}")
        ax.set_xlabel("T_u (number of observations per user)",
                      fontsize=13)
        ax.set_ylabel("Number of users", fontsize=13)
        ax.set_title(f"{algo} — user activity distribution",
                     fontsize=13)
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, axis='y', alpha=0.3)
        ax.tick_params(axis='both', labelsize=11)

        # --- Panel 3: cumulative recluster events over time ---
        ax = axes[2]
        cum = _cumulative_reclusters_over_time(user_stream, freq)
        ax.plot(np.arange(T), cum, color='#9467bd', linewidth=1.5)
        ax.set_xlabel("Iteration t", fontsize=13)
        ax.set_ylabel("Cumulative recluster events", fontsize=13)
        ax.set_title(f"{algo} — cumulative reclusters over time",
                     fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=11)
        # Annotation: how much is init vs post-init at end.
        ax.text(0.02, 0.95,
                f"final = {int(cum[-1])}\n"
                f"init = {n_users}\n"
                f"post-init = {total_post_init}",
                transform=ax.transAxes, ha='left', va='top',
                fontsize=10, bbox=dict(boxstyle='round,pad=0.4',
                                       facecolor='white', alpha=0.8))

        fig.suptitle(
            f"[{os.path.basename(exp_dir)}]  {algo}  "
            f"arm_recluster_freq={freq}", fontsize=14, y=1.01,
        )
        plt.tight_layout()
        out = os.path.join(exp_dir,
                            f"recluster_histogram.{algo}.pdf")
        fig.savefig(out, dpi=200, bbox_inches='tight', format='pdf')
        plt.close(fig)
        print(f"  saved {out}")


def main():
    ap = argparse.ArgumentParser(
        description="Reconstruct per-user recluster event counts from a "
                    "saved experiment (no rerun).",
    )
    ap.add_argument('exp_dir', help="Experiment directory (e.g. "
                                     "results/movielens_all).")
    ap.add_argument('--algo', nargs='+', required=True,
                    help="Algorithm name(s) (e.g. LinUCB_IND_RP_CA). "
                         "Only LinUCB_IND_* family handled by this version.")
    ap.add_argument('--arm-recluster-freq', type=int, required=True,
                    help="arm_recluster_freq value used by the algorithm "
                         "during the run. Must match what was used.")
    ap.add_argument('--user-stream', default=None,
                    help="Optional path to a user_stream .npy or .npz. "
                         "If omitted, the script reads _meta.npz "
                         "from the experiment directory.")
    args = ap.parse_args()

    if not os.path.isdir(args.exp_dir):
        print(f"ERROR: '{args.exp_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    plot_recluster_histogram(
        exp_dir=args.exp_dir,
        algos=args.algo,
        freq=args.arm_recluster_freq,
        user_stream_path=args.user_stream,
    )


if __name__ == '__main__':
    main()