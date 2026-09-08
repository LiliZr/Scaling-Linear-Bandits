"""
audit_cluster_data.py — verify that every clustering / arm-bucketing algorithm
actually saved its size evolution, and emit re-run commands for the ones that
did not.

Usage:
    python audit_cluster_data.py results/*

What it checks, per results directory and per algorithm:
  * n_user_clusters: expected to vary (> 1 at some point) for algorithms that
    cluster users -> USER_CLUSTERING_ALGOS, plus the periodic k-means variant
    (name contains 'KMeans'), which clusters users without an online `clusters`
    attribute.
  * n_arm_groups: expected to be > 0 for algorithms that bucket arms
    -> ARM_CLUSTERING_ALGOS.

The "all zero n_arm_groups" (or "constant n_user_clusters == 1") signature is
exactly what the pre-fix _track_state produced for the LinUCB_*_RP_CA family
(they own an `arm_clust` but no `clusters`). After patching base.py and
re-running those algos, this audit should report them clean.

For every directory with broken algos, a ready re-run command is printed,
filled with T / k / nb_runs read from _meta.npz. Only the data source line is
left to complete (real-data vs synthetic preset).
"""
import os
import sys

import numpy as np

try:
    from algo_registry import USER_CLUSTERING_ALGOS, ARM_CLUSTERING_ALGOS
except Exception:
    USER_CLUSTERING_ALGOS = {
        'CLUB', 'SCLUB', 'SCLUB_RP', 'SCLUB_CA', 'SCLUB_RP_CA',
        'Bandit_Diag', 'Bandit_Sketch', 'Bandit_History', 'Bandit_Window',
    }
    ARM_CLUSTERING_ALGOS = {
        'SCLUB_CA', 'SCLUB_RP_CA', 'Bandit_Diag', 'Bandit_Sketch',
        'Bandit_History', 'Bandit_Window', 'LinUCB_RP_CA',
        'LinUCB_IND_RP_CA', 'LinUCB_IND_RP_CA_KMeans',
    }


def expects_user_clustering(algo):
    return algo in USER_CLUSTERING_ALGOS or 'KMeans' in algo


def expects_arm_bucketing(algo):
    return algo in ARM_CLUSTERING_ALGOS


def _load_meta(results_dir):
    path = os.path.join(results_dir, '_meta.npz')
    if not os.path.exists(path):
        return {}
    d = np.load(path, allow_pickle=True)
    return {k: (d[k].item() if d[k].ndim == 0 else d[k]) for k in d.files}


def _slice_completed(arr, completed):
    if arr.ndim == 2 and completed and completed > 0:
        return arr[:completed]
    return arr


def audit_dir(results_dir):
    if not os.path.isdir(results_dir):
        return None
    meta = _load_meta(results_dir)
    rows = []
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith('.npz') or fname == '_meta.npz':
            continue
        algo = fname[:-4]
        try:
            data = np.load(os.path.join(results_dir, fname), allow_pickle=True)
        except Exception as e:
            print(f"  [warn] cannot read {fname}: {e}")
            continue
        keys = set(data.files)
        completed = int(data['completed_runs'].item()) if 'completed_runs' in keys else None

        exp_uc = expects_user_clustering(algo)
        exp_ag = expects_arm_bucketing(algo)

        uc_max = ag_max = None
        if 'n_user_clusters' in keys:
            uc_max = int(_slice_completed(data['n_user_clusters'], completed).max(initial=0))
        if 'n_arm_groups' in keys:
            ag_max = int(_slice_completed(data['n_arm_groups'], completed).max(initial=0))

        # broken = expected to track but the saved values are trivial
        broken_uc = exp_uc and (uc_max is None or uc_max <= 1)
        broken_ag = exp_ag and (ag_max is None or ag_max == 0)
        rows.append(dict(algo=algo, exp_uc=exp_uc, exp_ag=exp_ag,
                         uc_max=uc_max, ag_max=ag_max,
                         broken_uc=broken_uc, broken_ag=broken_ag))

    print(f"\n=== {results_dir} ===")
    if meta:
        print(f"  meta: T={meta.get('T')}, k={meta.get('k')}, "
              f"nb_runs={meta.get('nb_runs')}, n_items={meta.get('n_items')}")
    print(f"  {'algo':30s} {'n_user_clusters':>16s} {'n_arm_groups':>14s}   status")
    for r in rows:
        uc = "-" if not r['exp_uc'] else (f"max={r['uc_max']}")
        ag = "-" if not r['exp_ag'] else (f"max={r['ag_max']}")
        problems = []
        if r['broken_uc']:
            problems.append("user clusters NOT tracked")
        if r['broken_ag']:
            problems.append("arm groups NOT tracked")
        status = "BROKEN: " + "; ".join(problems) if problems else "ok"
        print(f"  {r['algo']:30s} {uc:>16s} {ag:>14s}   {status}")

    broken = sorted({r['algo'] for r in rows if r['broken_uc'] or r['broken_ag']})
    if broken:
        T = meta.get('T', '<T>')
        k = meta.get('k', '<k>')
        nb = meta.get('nb_runs', '<nb_runs>')
        algos = " ".join(broken)
        print(f"\n  -> RE-RUN (after patching base.py). Pick ONE data line:")
        print(f"     python main.py {algos} \\")
        print(f"         --out-dir {results_dir} --T {T} --k {k} --nb_runs {nb} --force \\")
        print(f"         --data-npz PATH/TO/your_dataset.npz      # real-data datasets")
        print(f"     # synthetic instead: drop --data-npz and add  --experiment <preset>  "
              f"(same preset/seed as the original run)")
    else:
        print("\n  -> nothing to re-run in this directory.")
    return broken


def main():
    if len(sys.argv) < 2:
        print("usage: python audit_cluster_data.py results/dir1 [results/dir2 ...]")
        sys.exit(1)
    summary = {}
    for d in sys.argv[1:]:
        res = audit_dir(d)
        if res is not None:
            summary[d] = res
    print("\n=== summary: directories needing a re-run ===")
    any_broken = False
    for d, b in summary.items():
        if b:
            any_broken = True
            print(f"  {d}: {' '.join(b)}")
    if not any_broken:
        print("  (none)")


if __name__ == "__main__":
    main()