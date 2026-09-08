"""
algo_registry.py — single source of truth for algorithms.

Contains:
  - ALGOS:   class, default params, "needs_k" flag (k taken from experiment).
  - STYLES:  fixed (color, marker) per algorithm name. Keep these stable across
             experiments so plots remain comparable.
  - CATEGORIES: which algorithms do user clustering / arm clustering.

Defaults follow the original papers wherever possible:
  - Random:  uniform random arm selection. Sanity-check baseline.
  - LinUCB / LinUCB_RP: only `lam` and `scale` (UCB exploration).
  - LinUCB_IND / LinUCB_IND_RP: same, one model per user.
  - CLUB:    `alpha_cb` (edge deletion CB), `scale` (UCB).
  - SCLUB:   `alpha_theta` (split/merge CB), `scale` (UCB), phase structure 2^s implicit.
  - SCLUB_RP, SCLUB_CA, SCLUB_RP_CA: same as SCLUB plus `k` for projection and/or
    `n_arm_groups`/`arm_recluster_freq` for arm clustering.
  - Bandit_*:    SCLUB_RP_CA + per-user state approximation.
"""
from algorithms.linucb.linucb import Random
from algorithms.linucb.linucb import LinUCB
from algorithms.linucb.linucb_rp import LinUCB_RP
from algorithms.linucb.linucb_ind import LinUCB_IND
from algorithms.linucb.linucb_ind_rp import LinUCB_IND_RP
from algorithms.linucb.linucb_ind_rp_ca import LinUCB_IND_RP_CA
from algorithms.linucb.linucb_ind_rp_ca_kmeans import LinUCB_IND_RP_CA_KMeans
from algorithms.club.club import CLUB
from algorithms.sclub.sclub import SCLUB
from algorithms.sclub.sclub_rp import SCLUB_RP
from algorithms.sclub.sclub_ca import SCLUB_CA
from algorithms.sclub.sclub_rp_ca import SCLUB_RP_CA
from algorithms.bandit.bandit_diag import Bandit_Diag
from algorithms.bandit.bandit_sketch import Bandit_Sketch
from algorithms.bandit.bandit_history import Bandit_History
from algorithms.bandit.bandit_window import Bandit_Window
from algorithms.linucb.linucb_rp_ca import LinUCB_RP_CA


# Default param dicts. Common params (`lam`, `scale`, `seed`) are filled in by the runner.
# 'needs_k': if True, the runner will pass k=experiment.k to the algorithm.
ALGOS = {
    # ---------- Sanity-check baseline ----------
    'Random': {
        'class': Random,
        'params': {},
        'needs_k': False,
    },
    # ---------- Baselines ----------
    'LinUCB': {
        'class': LinUCB,
        'params': {},
        'needs_k': False,
    },
    'LinUCB_RP': {
        'class': LinUCB_RP,
        'params': {},
        'needs_k': True,
    },
    'LinUCB_IND': {
        'class': LinUCB_IND,
        'params': {},
        'needs_k': False,
    },
    'LinUCB_IND_RP': {
        'class': LinUCB_IND_RP,
        'params': {},
        'needs_k': True,
    },
    'LinUCB_RP_CA': {
        'class': LinUCB_RP_CA,
        'params': {'arm_recluster_freq': 500},
        'needs_k': True,
    },
    'LinUCB_IND_RP_CA': {
        'class': LinUCB_IND_RP_CA,
        'params': {'arm_recluster_freq': 500},
        'needs_k': True,
    },
    'LinUCB_IND_RP_CA_KMeans': {
        'class': LinUCB_IND_RP_CA_KMeans,
        'params': {'arm_recluster_freq': 500,
                   'n_user_clusters': 5, 'kmeans_freq': 2000},
        'needs_k': True,
    },
    # ---------- Existing clustering algorithms ----------
    'CLUB': {
        'class': CLUB,
        'params': {'alpha_cb': 2.0},
        'needs_k': False,
    },
    'SCLUB': {
        'class': SCLUB,
        'params': {'alpha_theta': 1.0},
        'needs_k': False,
    },
    'SCLUB_RP': {
        # SCLUB + RP only. Same alpha_theta as SCLUB, no arm clustering.
        'class': SCLUB_RP,
        'params': {'alpha_theta': 1.0},
        'needs_k': True,
    },
    'SCLUB_CA': {
        # SCLUB + arm clustering, no RP. Same alpha_theta.
        'class': SCLUB_CA,
        'params': {'alpha_theta': 1.0},
        'needs_k': False,
    },
    'SCLUB_RP_CA': {
        # SCLUB + RP + arm clustering.
        'class': SCLUB_RP_CA,
        'params': {'alpha_theta': 1.0},
        'needs_k': True,
    },
    # ---------- Our proposals: SCLUB_RP_CA with per-user S_u approximation ----------
    'Bandit_Diag': {
        'class': Bandit_Diag,
        'params': {'alpha_theta': 1.0},
        'needs_k': True,
    },
    'Bandit_Sketch': {
        'class': Bandit_Sketch,
        'params': {'alpha_theta': 1.0, 'sketch_rank': 5},
        'needs_k': True,
    },
    'Bandit_History': {
        'class': Bandit_History,
        'params': {'alpha_theta': 1.0},
        'needs_k': True,
    },
    'Bandit_Window': {
        'class': Bandit_Window,
        'params': {'alpha_theta': 1.0},
        'needs_k': True,
    },
}


# Fixed style per algorithm. Colors chosen distinctively; markers add a second
# disambiguating channel.
STYLES = {
    'Random':         ('#999999', 'x'),
    'LinUCB':         ('#1f77b4', 'o'),
    'LinUCB_RP':      ('#17becf', 's'),
    'LinUCB_IND':     ('#2ca02c', '^'),
    'LinUCB_IND_RP':  ('#98df8a', 'v'),
    'LinUCB_IND_RP_CA': ('#f7b6d2', 'p'),
    'LinUCB_IND_RP_CA_KMeans': ('#c49c94', '8'),
    'CLUB':           ('#8c564b', '<'),
    'SCLUB':          ('#d62728', 'D'),
    'SCLUB_RP':       ('#ff7f0e', 'P'),
    'SCLUB_CA':       ('#9467bd', 'X'),
    'SCLUB_RP_CA':    ('#e377c2', '*'),
    'Bandit_Diag':    ('#000000', 'H'),
    'Bandit_Sketch':  ('#bcbd22', 'h'),
    'Bandit_History': ('#7f7f7f', 'p'),
    'Bandit_Window':  ('#aec7e8', 'd'),
    'LinUCB_RP_CA': ('#9edae5', 'P'),
}

# Algorithms that perform user clustering (relevant for cluster-evolution plots).
USER_CLUSTERING_ALGOS = {
    'CLUB', 'SCLUB', 'SCLUB_RP', 'SCLUB_CA', 'SCLUB_RP_CA',
    'Bandit_Diag', 'Bandit_Sketch', 'Bandit_History', 'Bandit_Window',
}

# Algorithms that perform arm clustering.
ARM_CLUSTERING_ALGOS = {
    'SCLUB_CA', 'SCLUB_RP_CA',
   # 'Bandit_Diag', 'Bandit_Sketch', 'Bandit_History', 'Bandit_Window',
    'LinUCB_RP_CA',                      
    'LinUCB_IND_RP_CA', 'LinUCB_IND_RP_CA_KMeans',
}

# Named GROUPS of algorithms for convenient launches.
# Pick a server, name a group, run it. Edit freely.
#
# Suggested split for your setup:
#   server1 (slow group)  : SCLUB + LinUCB-family + CLUB + Random
#   server2 (RP group)    : LinUCB_RP, LinUCB_IND_RP, SCLUB_RP, SCLUB_RP_CA, SCLUB_CA
#   server3 (Bandit group): Bandit_Diag, Bandit_Sketch, Bandit_History, Bandit_Window
GROUPS = {
    # ----- The split you described -----
    'slow':     ['Random', 'LinUCB', 'LinUCB_IND', 'CLUB', 'SCLUB'],
    'rd':       ['LinUCB_RP', 'LinUCB_IND_RP', 'LinUCB_RP_CA', 'LinUCB_IND_RP_CA', 'LinUCB_IND_RP_CA_KMeans', 'SCLUB_RP', 'SCLUB_CA', 'SCLUB_RP_CA', ],
    'bandits':  ['Bandit_History', 'Bandit_Window', 'Bandit_Sketch',  'Bandit_Diag',],
    # ----- Smaller pairs / triples if you want finer-grained partitioning -----
    'lin':      ['LinUCB', 'LinUCB_RP'],
    'lin_ind':  ['LinUCB_IND', 'LinUCB_IND_RP'],
    'sclub_no_rp':  ['SCLUB', 'SCLUB_CA'],
    'sclub_rp':     ['SCLUB_RP', 'SCLUB_RP_CA'],
    'bandit_lite':  ['Bandit_Diag', 'Bandit_Window'],
    'bandit_heavy': ['Bandit_Sketch', 'Bandit_History'],
    # ----- Sanity check (Random alone, useful to gauge the absolute scale) -----
    'random':   ['Random'],
    # ----- Catch-all -----
    'all':      list(ALGOS.keys()),
}


def resolve_targets(tokens):
    """Expand a list of tokens (algo names OR group names) into algo names.

    Tokens are case-insensitive. Unknowns raise ValueError listing valid options.
    """
    out = []
    seen = set()
    valid_algos = {a.lower(): a for a in ALGOS}
    valid_groups = {g.lower(): g for g in GROUPS}
    for tok in tokens:
        key = tok.lower()
        if key in valid_groups:
            for a in GROUPS[valid_groups[key]]:
                if a not in seen:
                    seen.add(a)
                    out.append(a)
        elif key in valid_algos:
            a = valid_algos[key]
            if a not in seen:
                seen.add(a)
                out.append(a)
        else:
            raise ValueError(
                f"Unknown target '{tok}'. "
                f"Algorithms: {list(ALGOS.keys())}. "
                f"Groups: {list(GROUPS.keys())}."
            )
    return out


def build_model(algo_name, data, weights, seed, k, lam=1.0, scale=0.5,
                param_overrides=None, batch=False, batch_size=20,
                ucb_partition=False,
                update_champions_on_select=False,
                track_bucket_selection=False):
    """Instantiate an algorithm by name with the right params filled in.

    The three trailing flags only apply to algorithms in ARM_CLUSTERING_ALGOS
    (they are forwarded to ArmClustering via the algo constructor). For all
    other algorithms the flags are silently ignored, so it is safe to pass
    them on a generic launch.
    """
    if algo_name not in ALGOS:
        raise ValueError(f"Unknown algorithm: {algo_name}. Choices: {list(ALGOS.keys())}")
    spec = ALGOS[algo_name]
    cls = spec['class']
    params = dict(spec['params'])
    if spec['needs_k']:
        params['k'] = k
    params['lam'] = lam
    params['scale'] = scale
    params['batch'] = batch
    params['batch_size'] = batch_size
    # Forward arm-clustering flags only to algorithms that support them.
    if algo_name in ARM_CLUSTERING_ALGOS:
        params['ucb_partition'] = bool(ucb_partition)
        params['update_champions_on_select'] = bool(update_champions_on_select)
        params['track_bucket_selection'] = bool(track_bucket_selection)
    if param_overrides and algo_name in param_overrides:
        params.update(param_overrides[algo_name])
    return cls(data=data, weights=weights, seed=seed, **params)