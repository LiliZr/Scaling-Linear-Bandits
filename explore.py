"""
explore.py -- Diagnostic experiments for LinUCB vs LinUCB_IND, and RP.

Two questions:
  1. When does LinUCB beat LinUCB_IND? Controlled by the SIMILARITY of
     users (parameter frac_similar_users) and the AVAILABLE DATA per user
     (T / n_users).
  2. When does random projection help?  Controlled by the EFFECTIVE
     dimensionality of the features.

Usage:
  python explore.py --test B            # similar users + cold-start
  python explore.py --test D            # distinct users + plenty of data
  python explore.py --test RP_HELPS     # high-d, RP regularises noise
  python explore.py --test RP_NEUTRAL   # low-d, no benefit from RP

  python explore.py --test B --frac-similar 0.5  # 50% of users in 1 cluster
                                                 # 50% with their own theta
  python explore.py --test B --T 30000 --n-seeds 15
"""
import argparse
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.synthetic import generate_user_stream
from algorithms.linucb.linucb import LinUCB
from algorithms.linucb.linucb_rp import LinUCB_RP
from algorithms.linucb.linucb_ind import LinUCB_IND
from algorithms.linucb.linucb_ind_rp import LinUCB_IND_RP


# Plot style: matches the reference style used elsewhere in the codebase.
STYLE = {
    'LinUCB':         {'color': '#1f77b4', 'marker': 'o', 'label': 'LinUCB'},
    'LinUCB_RP':      {'color': '#17becf', 'marker': 's', 'label': 'LinUCB_RP'},
    'LinUCB_IND':     {'color': '#2ca02c', 'marker': '^', 'label': 'LinUCB_IND'},
    'LinUCB_IND_RP':  {'color': '#98df8a', 'marker': 'v', 'label': 'LinUCB_IND_RP'},
}


# ---------------------------------------------------------------------------
# Synthetic data with controllable similarity between users
# ---------------------------------------------------------------------------

def make_data(n_users, n_items, d, frac_similar_users, D_objectives,
              noise_std, seed, intra_cluster_spread=0.01):
    """Generate a bandit instance where a fraction `frac_similar_users` of
    users have preference vectors NEAR a common anchor, and the remaining
    users each have their own independently drawn theta.

    Parameters
    ----------
    frac_similar_users : float in [0, 1]
        Fraction of users sharing the anchor cluster.
        1.0 -> all users near the anchor (B-style)
        0.0 -> every user has an independent theta (D-style)
        0.5 -> half near the anchor, half independent
    intra_cluster_spread : float, default 0.01
        Standard deviation of the Gaussian perturbation applied around
        the anchor for similar users. Controls HOW similar they are:
          0.0   -> all similar users have EXACTLY the same theta (identical)
          0.01  -> very close: ||theta_u - anchor|| ~ 0.07 in d=50
                              (i.e. ~7% of the anchor's norm, the default)
          0.1   -> close-ish: deviation ~ 70% of anchor norm
          0.5   -> as spread out as the independent users (no cluster anymore)
    """
    rng = np.random.RandomState(seed)

    # ----- arm features -----
    n_arm_clusters = 20
    arm_centers = rng.randn(n_arm_clusters, d).astype(np.float32)
    arm_centers /= np.linalg.norm(arm_centers, axis=1, keepdims=True)
    arm_assign = rng.randint(0, n_arm_clusters, size=n_items)
    item_features = np.zeros((n_items, d), dtype=np.float32)
    for a in range(n_items):
        item_features[a] = (arm_centers[arm_assign[a]]
                            + rng.randn(d).astype(np.float32) * 0.1)
    item_features /= np.linalg.norm(item_features, axis=1, keepdims=True)

    # ----- user thetas -----
    # Anchor cluster theta (centroid of the "similar" users)
    anchor = rng.randn(D_objectives, d).astype(np.float32) * 0.5
    anchor /= np.linalg.norm(anchor, axis=1, keepdims=True)

    n_similar = int(frac_similar_users * n_users)
    user_thetas = {}
    for u in range(n_users):
        if u < n_similar:
            # Similar users: anchor + Gaussian perturbation.
            # intra_cluster_spread = 0.0 makes them all IDENTICAL.
            user_thetas[u] = (
                anchor
                + rng.randn(D_objectives, d).astype(np.float32)
                  * intra_cluster_spread
            )
        else:
            theta = rng.randn(D_objectives, d).astype(np.float32) * 0.5
            theta /= np.linalg.norm(theta, axis=1, keepdims=True)
            user_thetas[u] = theta

    return {
        'item_features': item_features,
        'user_thetas':   user_thetas,
        'n_users':       n_users,
        'n_items':       n_items,
        'd':             d,
        'D_objectives':  D_objectives,
        'noise_std':     noise_std,
    }


def add_noise_dimensions(data, n_extra_dims, seed):
    """Append `n_extra_dims` columns of pure noise to item features (with
    zero in the corresponding rows of user_thetas, so they carry no signal).
    Increases the effective feature dimension d_eff = d + n_extra_dims.
    """
    rng = np.random.RandomState(seed)
    X = data['item_features']
    K = X.shape[0]
    noise = rng.randn(K, n_extra_dims).astype(np.float32) * 0.3
    X_new = np.hstack([X, noise])
    ut_new = {}
    for u, th in data['user_thetas'].items():
        D = th.shape[0]
        ut_new[u] = np.concatenate(
            [th, np.zeros((D, n_extra_dims), dtype=th.dtype)], axis=1
        )
    new = dict(data)
    new['item_features'] = X_new
    new['user_thetas'] = ut_new
    new['d'] = X_new.shape[1]
    return new


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_one_test(name, params, out_dir):
    """Run all 4 algorithms with `n_seeds` random seeds and plot the mean
    cumulative regret. Returns a dict of final regrets."""
    n_seeds = params['n_seeds']
    T = params['T']
    print(f"\n=== Test {name} ===")
    print(f"  params: {params}")

    regrets = {cls.__name__: np.zeros((n_seeds, T)) for cls in
               [LinUCB, LinUCB_RP, LinUCB_IND, LinUCB_IND_RP]}

    for s_idx, seed in enumerate(range(42, 42 + n_seeds)):
        data = make_data(
            n_users=params['n_users'], n_items=params['n_items'],
            d=params['d'],
            frac_similar_users=params['frac_similar_users'],
            D_objectives=2,
            noise_std=params['noise_std'], seed=seed,
            intra_cluster_spread=params.get('intra_cluster_spread', 0.01),
        )
        if params.get('extra_noise_dims', 0) > 0:
            data = add_noise_dimensions(
                data, params['extra_noise_dims'], seed=seed)
        # Stream: uniform unless --skew specified
        us = generate_user_stream(
            params['n_users'], T, seed=seed + 1000,
            skew=params.get('skew', 'uniform'),
            heavy_frac=params.get('heavy_frac', 0.1),
            heavy_mass=params.get('heavy_mass', 0.85),
        )
        for cls in [LinUCB, LinUCB_RP, LinUCB_IND, LinUCB_IND_RP]:
            kw = {'k': params['k']} if cls.__name__.endswith('RP') else {}
            m = cls(data=data, weights=np.ones(2) / 2,
                    lam=1.0, scale=0.5, seed=seed, **kw)
            m.run(us, T)
            regrets[cls.__name__][s_idx] = m.cumulative_regret

    # -------- plot --------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(T)
    markevery = max(1, T // 20)
    for n in ['LinUCB', 'LinUCB_RP', 'LinUCB_IND', 'LinUCB_IND_RP']:
        mean = regrets[n].mean(axis=0)
        s = STYLE[n]
        ax.plot(x, mean, color=s['color'], marker=s['marker'],
                markevery=markevery, markersize=6, linewidth=1.5,
                label=s['label'])
    d_eff = params['d'] + params.get('extra_noise_dims', 0)
    spread = params.get('intra_cluster_spread', 0.01)
    sub = (f"n_users={params['n_users']}, n_items={params['n_items']}, "
           f"d={d_eff}, k={params['k']}, T={T}, "
           f"frac_similar={params['frac_similar_users']:.2f} "
           f"(spread={spread:g}), n_seeds={n_seeds}")
    ax.set_title(f"Test {name}\n{sub}", fontsize=10)
    ax.set_xlabel('Round')
    ax.set_ylabel('Cumulative Regret')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'scenario_{name}.png')
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {out_path}")

    print(f"  final regrets (mean ± std):")
    for n in ['LinUCB', 'LinUCB_RP', 'LinUCB_IND', 'LinUCB_IND_RP']:
        arr = regrets[n][:, -1]
        print(f"    {n:<18} {arr.mean():8.1f} ± {arr.std():.1f}")
    return regrets


# ---------------------------------------------------------------------------
# Pre-configured tests
# ---------------------------------------------------------------------------

TESTS = {
    # ---- LinUCB vs LinUCB_IND ----
    'B': dict(
        n_users=2000, n_items=500, d=50, k=10,
        T=15000, n_seeds=15,
        frac_similar_users=1.0,                  # all users in 1 cluster
        noise_std=0.05, skew='uniform',
    ),
    'C': dict(  # like B but with skewed stream
        n_users=2000, n_items=500, d=50, k=10,
        T=15000, n_seeds=15,
        frac_similar_users=1.0,
        noise_std=0.05, skew='mixed',
        heavy_frac=0.1, heavy_mass=0.85,
    ),
    'D': dict(
        n_users=50, n_items=500, d=50, k=10,
        T=20000, n_seeds=20,
        frac_similar_users=0.0,                  # all distinct
        noise_std=0.05, skew='uniform',
    ),

    # ---- Random projection ----
    'RP_HELPS': dict(  # large effective dimension -> RP regularises noise
        n_users=100, n_items=500, d=10, k=20,
        T=8000, n_seeds=10,
        frac_similar_users=0.0,
        noise_std=0.1, skew='uniform',
        extra_noise_dims=200,                    # d_eff = 10 + 200 = 210
    ),
    'RP_NEUTRAL': dict(  # small d, no noise -> RP cannot help
        n_users=3000, n_items=500, d=8, k=4,
        T=20000, n_seeds=20,
        frac_similar_users=1.0,
        noise_std=0.005, skew='uniform',
    ),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--test', required=True,
                   choices=list(TESTS.keys()),
                   help='Pre-configured test to run.')
    p.add_argument('--frac-similar', type=float, default=None,
                   help='Override frac_similar_users for this test '
                        '(fraction of users near the anchor).')
    p.add_argument('--intra-spread', type=float, default=None,
                   help='Override intra_cluster_spread: std of the noise '
                        'around the anchor. 0.0 = users identical, '
                        '0.01 = very close (default), 0.5 = essentially '
                        'distinct.')
    p.add_argument('--T', type=int, default=None,
                   help='Override the time horizon T.')
    p.add_argument('--n-seeds', type=int, default=None,
                   help='Override the number of seeds.')
    p.add_argument('--n-users', type=int, default=None,
                   help='Override the number of users.')
    p.add_argument('--d', type=int, default=None,
                   help='Override the feature dimension d.')
    p.add_argument('--k', type=int, default=None,
                   help='Override the projection dimension k.')
    p.add_argument('--out-dir', default='exploration_plots',
                   help='Output directory for plots.')
    args = p.parse_args()

    params = dict(TESTS[args.test])
    if args.frac_similar is not None: params['frac_similar_users'] = args.frac_similar
    if args.intra_spread is not None: params['intra_cluster_spread'] = args.intra_spread
    if args.T is not None: params['T'] = args.T
    if args.n_seeds is not None: params['n_seeds'] = args.n_seeds
    if args.n_users is not None: params['n_users'] = args.n_users
    if args.d is not None: params['d'] = args.d
    if args.k is not None: params['k'] = args.k

    t0 = time.time()
    run_one_test(args.test, params, args.out_dir)
    print(f"  total time: {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()