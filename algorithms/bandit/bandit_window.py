"""
Bandit_Window: SCLUB_RP_CA with garbage collection of inactive users.

EVERYTHING ELSE IS IDENTICAL to SCLUB_RP_CA. The user-level stats S_u, b_u, etc.
are kept exactly as in SCLUB.

THE ONLY ADDITION:
  Users that have not been served for more than `window_size` rounds have their
  per-user state (S_u, S_u_inv, b_u, theta_u) released. Their cluster membership
  is preserved. If they reappear, they are re-initialized from scratch (warm
  start would require keeping a small summary, which we don't, to keep memory
  truly minimal).

  No splits or merges are triggered by GC: the cluster's aggregate (S, b) is
  unchanged. Only the user-level cache is dropped.

This variant is most useful when the user stream is sparse (some users rarely
return), in which case memory grows linearly with active users instead of total
users.

Hyperparameter:
  window_size: maximum gap (in rounds) between two visits before eviction.
  gc_freq: how often the GC scan runs (default 1000).
"""
import numpy as np
from algorithms.sclub.sclub_rp_ca import SCLUB_RP_CA


class Bandit_Window(SCLUB_RP_CA):
    def __init__(self, data, weights, k=20, lam=1.0, scale=1.0, seed=42,
                 alpha_theta=1.0, seed_proj=1,
                 n_arm_groups=None, arm_recluster_freq=500,
                 window_size=None, gc_freq=1000,
                 batch=False, batch_size=20, split_merge_freq=1):
        super().__init__(data, weights, k=k, lam=lam, scale=scale, seed=seed,
                         alpha_theta=alpha_theta, seed_proj=seed_proj,
                         n_arm_groups=n_arm_groups,
                         arm_recluster_freq=arm_recluster_freq,
                         batch=batch, batch_size=batch_size, split_merge_freq=split_merge_freq)
        # Default: 10x #users (each user visited on average every n rounds).
        self.window_size = window_size if window_size is not None else 10 * len(self.user_thetas)
        self.gc_freq = gc_freq

    def init_run(self, T):
        super().init_run(T)
        self.last_seen = {}
        self.n_evictions = 0

    def _init_user(self, user_id):
        super()._init_user(user_id)
        self.last_seen[user_id] = 0

    def update(self, user_id, item_id, reward_vec, reward_scalar, t):
        # If the user was evicted, re-initialize their state from scratch.
        if user_id not in self.S_u:
            self.S_u[user_id] = np.eye(self.dim)
            self.S_u_inv[user_id] = np.eye(self.dim)
            self.b_u[user_id] = np.zeros(self.dim)
            self.theta_u[user_id] = np.zeros(self.dim)
            self.T_u[user_id] = 0
        self.last_seen[user_id] = t

        super().update(user_id, item_id, reward_vec, reward_scalar, t)

        if (t + 1) % self.gc_freq == 0:
            self._garbage_collect(t)

    def _garbage_collect(self, t):
        to_evict = [u for u, last in self.last_seen.items()
                    if t - last > self.window_size and u in self.S_u]
        for u in to_evict:
            self.S_u.pop(u, None)
            self.S_u_inv.pop(u, None)
            self.b_u.pop(u, None)
            self.theta_u.pop(u, None)
            self.T_u.pop(u, None)
            self.n_evictions += 1

    def _try_split(self, user_id, t):
        # Cannot split a user whose state was evicted.
        if user_id not in self.T_u or self.T_u[user_id] < 2:
            return
        super()._try_split(user_id, t)
