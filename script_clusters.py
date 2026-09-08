"""
freq_sweep.py — sweep over user-cluster vs arm-cluster recluster frequencies
to measure their impact on regret/CPU/memory.

Two experiments are defined:

  EXP1 (vary user recluster frequency, arm recluster frequency held fixed):
    For each algo in {SCLUB, SCLUB_RP, SCLUB_CA, SCLUB_RP_CA},
    run with arm_recluster_freq = ARM_FIXED and split_merge_freq in
    {1, 100, 500, 1000, 2000}.

  EXP2 (vary arm recluster frequency, user recluster frequency held fixed):
    For each algo with arm clustering in {SCLUB_CA, SCLUB_RP_CA},
    run with split_merge_freq = USER_FIXED and arm_recluster_freq in
    {1, 100, 200, 500, 1000}.

When the per-cluster arm-recluster counter is enabled (current default in
sclub_ca.py / sclub_rp_ca.py after the per-cluster fix), the arm-recluster
frequency should be divided by an approximate number of users per cluster.
Concretely:
  ARM_FIXED:   500 (legacy global) -> 50 (per-cluster equivalent)
  EXP2 grid:   {1, 100, 200, 500, 1000} (legacy) -> {1, 10, 20, 50, 100}

Set PER_CLUSTER_SEMANTICS = True (default) to use the divided values.

Usage:
    python freq_sweep.py --data-npz data/movielens_filtered.npz --T 150000 --nb-runs 3

Output:
    Each run goes to results/freq_sweep_exp{1,2}_{algo}_{freq}_<dataset>/
    A "freq_sweep_exp{1,2}_<algo>_summary.pdf" is produced as soon as
    enough data is available.
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

# Set to True if the per-cluster arm-recluster fix is active (sclub_ca and
# sclub_rp_ca count updates PER CLUSTER, so the same wallclock effect needs
# ~10x smaller numbers compared to the previous global-counter version).
PER_CLUSTER_SEMANTICS = True

if PER_CLUSTER_SEMANTICS:
    ARM_FIXED        = 50           # exp1: held fixed
    USER_FIXED       = 2000         # exp2: held fixed
    EXP1_USER_FREQS  = [1, 100, 500, 1000, 2000]
    EXP2_ARM_FREQS   = [1, 10, 20, 50, 100]
else:
    ARM_FIXED        = 500
    USER_FIXED       = 2000
    EXP1_USER_FREQS  = [1, 100, 500, 1000, 2000]
    EXP2_ARM_FREQS   = [1, 100, 200, 500, 1000]

EXP1_ALGOS = ['SCLUB', 'SCLUB_RP', 'SCLUB_CA', 'SCLUB_RP_CA']
EXP2_ALGOS = ['SCLUB_CA', 'SCLUB_RP_CA']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exp_dir(exp_tag, algo, freq, dataset_tag):
    return f"results/freq_sweep_{exp_tag}_{algo}_{freq}_{dataset_tag}"


def _load_npz_for(dir_path, algo):
    """Return the regret/cpu/memory mean curves and the T value, or None."""
    npz = os.path.join(dir_path, f"{algo}.npz")
    if not os.path.exists(npz):
        return None
    try:
        d = np.load(npz)
    except Exception:
        return None
    completed = int(d['completed_runs'].item()) if 'completed_runs' in d.files else 0
    if completed == 0:
        return None
    return {
        'regret':  d['regret'][:completed].mean(axis=0),
        'cpu':     d['cpu_time'][:completed].mean(axis=0),
        'memory':  d['memory'][:completed].mean(axis=0),
        'T':       d['regret'].shape[1],
        'completed': completed,
    }


def plot_sweep(exp_tag, algo, freq_grid, freq_label, dataset_tag, out_dir):
    """Generate one figure per algo with one curve per freq value.

    Saves PDF immediately so the user can inspect partial results while
    runs are still in progress.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(freq_grid)))
    any_curve = False
    for color, freq in zip(cmap, freq_grid):
        dirp = _exp_dir(exp_tag, algo, freq, dataset_tag)
        data = _load_npz_for(dirp, algo)
        if data is None:
            continue
        any_curve = True
        x = np.arange(data['T'])
        label = f"{freq_label}={freq} ({data['completed']} runs)"
        axes[0].plot(x, data['regret'], color=color, linewidth=1.6, label=label)
        axes[1].plot(x, np.clip(data['cpu'], 1e-4, None),
                     color=color, linewidth=1.6, label=label)
        axes[2].plot(x, np.clip(data['memory'], 1e-4, None),
                     color=color, linewidth=1.6, label=label)
    if not any_curve:
        plt.close(fig)
        return None
    titles  = ("Cumulative regret", "CPU time", "Memory")
    ylabels = ("Cumulative regret", "CPU time (s)", "Memory (MB)")
    for ax, title, ylab in zip(axes, titles, ylabels):
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Iteration", fontsize=13)
        ax.set_ylabel(ylab, fontsize=13)
        ax.tick_params(axis='both', labelsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best')
    axes[1].set_yscale('log')
    axes[2].set_yscale('log')
    fig.suptitle(f"[{exp_tag}] {algo} — {freq_label} sweep ({dataset_tag})",
                 fontsize=15)
    plt.tight_layout()
    out_path = os.path.join(out_dir,
                            f"freq_sweep_{exp_tag}_{algo}_summary.pdf")
    fig.savefig(out_path, dpi=200, bbox_inches='tight', format='pdf')
    plt.close(fig)
    return out_path


def run_one(algo, freq_user, freq_arm, exp_tag, freq_value, dataset_tag,
            args):
    """Launch a single experiment via main.py as a subprocess.

    The subprocess approach reuses all the bookkeeping (checkpoints,
    parallelism, row compaction) without duplicating logic. Each run lives
    in its own results directory so plotting can pick up results
    incrementally.
    """
    out_dir = _exp_dir(exp_tag, algo, freq_value, dataset_tag)
    os.makedirs(out_dir, exist_ok=True)

    # Optionally skip if checkpoint complete.
    existing = _load_npz_for(out_dir, algo)
    if existing is not None and existing['completed'] >= args.nb_runs:
        print(f"  [skip] {algo} {exp_tag} {freq_value}: already complete "
              f"({existing['completed']}/{args.nb_runs})")
        return

    # The split-merge-freq controls SCLUB's user-cluster test frequency.
    # arm_recluster_freq is a SCLUB_*_CA constructor argument; the only way
    # to override it from CLI is via the param-overrides JSON, which we
    # encode through environment variables read by main.py.
    cmd = [
        sys.executable, "main.py", algo,
        "--data-npz", args.data_npz,
        "--experiment", f"freq_sweep_{exp_tag}_{algo}_{freq_value}",
        "--T", str(args.T),
        "--nb_runs", str(args.nb_runs),
        "--n-workers", str(args.n_workers),
        "--split-merge-freq", str(freq_user),
        "--arm-recluster-freq", str(freq_arm),
        "--out-dir", out_dir,
    ]

    print(f"  [run]  {algo} {exp_tag} user_freq={freq_user} arm_freq={freq_arm}")
    t0 = time.time()
    rc = subprocess.call(cmd)
    dt = time.time() - t0
    print(f"  [done] rc={rc} elapsed={dt:.0f}s")


# ---------------------------------------------------------------------------
# Main sweep driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Frequency sweep over user/arm recluster periods.",
    )
    ap.add_argument('--data-npz', required=True,
                    help="Path to the preprocessed dataset .npz "
                         "(e.g. data/movielens_filtered.npz).")
    ap.add_argument('--T', type=int, default=150000,
                    help="Horizon (default 150000).")
    ap.add_argument('--nb-runs', type=int, default=3,
                    help="Number of seeds per (algo, freq).")
    ap.add_argument('--n-workers', type=int, default=0,
                    help="Parallel workers per run (default 4).")
    ap.add_argument('--out-dir', default='results/comparison_freq/',
                    help="Where to write the summary PDFs.")
    ap.add_argument('--exp', choices=['1', '2', 'both'], default='both')
    args = ap.parse_args()

    dataset_tag = os.path.splitext(os.path.basename(args.data_npz))[0]
    summary_dir = args.out_dir
    os.makedirs(summary_dir, exist_ok=True)

    # ---- Experiment 1: vary user recluster freq ----
    if args.exp in ('1', 'both'):
        print("\n=== Experiment 1: vary user recluster frequency "
              f"(arm fixed at {ARM_FIXED}) ===\n")
        for algo in EXP1_ALGOS:
            for f_user in EXP1_USER_FREQS:
                run_one(
                    algo=algo,
                    freq_user=f_user,
                    freq_arm=ARM_FIXED,
                    exp_tag='exp1',
                    freq_value=f_user,
                    dataset_tag=dataset_tag,
                    args=args,
                )
                # Plot after each run so partial progress is visible.
                plot_sweep('exp1', algo, EXP1_USER_FREQS,
                           freq_label='user_freq',
                           dataset_tag=dataset_tag,
                           out_dir=summary_dir)

    # ---- Experiment 2: vary arm recluster freq ----
    if args.exp in ('2', 'both'):
        print("\n=== Experiment 2: vary arm recluster frequency "
              f"(user fixed at {USER_FIXED}) ===\n")
        for algo in EXP2_ALGOS:
            for f_arm in EXP2_ARM_FREQS:
                run_one(
                    algo=algo,
                    freq_user=USER_FIXED,
                    freq_arm=f_arm,
                    exp_tag='exp2',
                    freq_value=f_arm,
                    dataset_tag=dataset_tag,
                    args=args,
                )
                plot_sweep('exp2', algo, EXP2_ARM_FREQS,
                           freq_label='arm_freq',
                           dataset_tag=dataset_tag,
                           out_dir=summary_dir)


if __name__ == '__main__':
    main()