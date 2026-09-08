"""
Bandit_History: SCLUB_RP_CA where the per-user state is reduced to a capped
sliding window of (item_id, reward_scalar) pairs.

EVERYTHING ELSE IS IDENTICAL to SCLUB_RP_CA (and SCLUB):
  - Same split/merge tests
  - Same F(T), phase structure, pivot snapshots
  - Same RP and arm clustering

THE ONE APPROXIMATION:
  Per-user state: list of (item_id, reward) tuples, capped at `max_history`.
  When the window saturates, oldest entries are dropped (FIFO via deque).
  S_u and b_u are recomputed on demand from the current window.

When a user is split out, S_u and b_u are reconstructed from their window;
this is exact within the window, approximate if T_u > max_history.

Memory per user: O(max_history) tuples (constant integers + floats), much
smaller than O(k²) when max_history is small.
"""
import numpy as np
from collections import deque
from algorithms.sclub.sclub_rp_ca import SCLUB_RP_CA
from utils.utils import inv_sherman_morrison


class Bandit_History(SCLUB_RP_CA):
    def __init__(self, data, weights, k=20, lam=1.0, scale=1.0, seed=42,
                 alpha_theta=1.0, seed_proj=1,
                 n_arm_groups=None, arm_recluster_freq=500,
                 max_history=None,
                 batch=False, batch_size=20, split_merge_freq=1):
        super().__init__(data, weights, k=k, lam=lam, scale=scale, seed=seed,
                         alpha_theta=alpha_theta, seed_proj=seed_proj,
                         n_arm_groups=n_arm_groups,
                         arm_recluster_freq=arm_recluster_freq,
                         batch=batch, batch_size=batch_size, split_merge_freq=split_merge_freq)
        self.max_history = max_history if max_history is not None else 4 * k

    def init_run(self, T):
        super().init_run(T)
        self.S_u = {}
        self.S_u_inv = {}
        self.history = {}  # user → deque of (item_id, reward_scalar)

    def _init_user(self, user_id):
        self.history[user_id] = deque(maxlen=self.max_history)
        self.b_u[user_id] = np.zeros(self.dim)
        self.theta_u[user_id] = np.zeros(self.dim)
        self.T_u[user_id] = 0
        best = max(self.clusters.keys(), key=lambda c: len(self.clusters[c]['C']))
        self.user_cluster[user_id] = best
        self.clusters[best]['C'].add(user_id)

    def _user_stats(self, user_id):
        """Reconstruct (S_u, b_u, θ̂_u) from the current window."""
        S = np.eye(self.dim)
        b = np.zeros(self.dim)
        for item_id, r in self.history[user_id]:
            z = self.X[item_id]
            S += np.outer(z, z)
            b += r * z
        theta = np.linalg.solve(S, b)
        return S, b, theta

    def recommend(self, user_id, t):
        if user_id not in self.history:
            self._init_user(user_id)
        self._check_phase_transition(t)
        c = self.user_cluster[user_id]
        cl = self.clusters[c]
        n_items = self.X.shape[0]
        beta = (np.sqrt(2 * np.log(2 * n_items * (t+1) / self.delta)) + np.sqrt(self.lam))
        return self.arm_clust.select(c, self.X, cl['theta_hat'], cl['S_inv'], beta)

    def update(self, user_id, item_id, reward_vec, reward_scalar, t):
        c = self.user_cluster[user_id]
        cl = self.clusters[c]
        x = self.X[item_id]
        xx = np.outer(x, x)

        # Append to capped window.
        self.history[user_id].append((item_id, reward_scalar))
        # Recompute b_u and theta_u from the window.
        # (Cheap: O(|window|) per update.)
        b_full = np.zeros(self.dim)
        for it, r in self.history[user_id]:
            b_full += r * self.X[it]
        self.b_u[user_id] = b_full
        self.T_u[user_id] += 1

        cl['S'] += xx
        inv_sherman_morrison(cl['S_inv'], x)
        cl['b'] += reward_scalar * x
        cl['theta_hat'] = cl['S_inv'] @ cl['b']
        cl['T'] += 1

        if (t + 1) % self.split_merge_freq == 0:
            source_cluster = self.user_cluster[user_id]
            self._try_split(user_id, t)
            self.user_checked.add(user_id)
            # Update checked_cids incrementally (see SCLUB.update for rationale).
            for c_check in {source_cluster, self.user_cluster[user_id]}:
                if c_check not in self.clusters:
                    continue
                if c_check in self.checked_cids:
                    continue
                cluster_users = self.clusters[c_check]['C']
                if cluster_users and cluster_users.issubset(self.user_checked):
                    self.checked_cids.add(c_check)
            if len(self.checked_cids) >= 2:
                self._try_merge(t)
        else:
            self.user_checked.add(user_id)

        if (t + 1) % self.arm_recluster_freq == 0:
            for cid, cluster in self.clusters.items():
                self.arm_clust.recluster(cid, self.X, cluster['theta_hat'])

    def _try_split(self, user_id, t):
        if self.T_u[user_id] < 2:
            return
        c = self.user_cluster[user_id]
        cl = self.clusters[c]
        theta_pivot = self.pivot_theta.get(c, cl['theta_hat'])
        T_pivot = self.pivot_T.get(c, cl['T'])
        T_i = self.T_u[user_id]

        # Reconstruct user stats from window (lazy, only on split test).
        S_u, b_u, theta_i = self._user_stats(user_id)
        self.theta_u[user_id] = theta_i

        threshold = self.alpha_theta * (self._F(T_i) + self._F(T_pivot))
        if np.linalg.norm(theta_i - theta_pivot) <= threshold:
            return

        new_id = self.next_cluster_id
        self.next_cluster_id += 1
        self._create_cluster(new_id)
        new_cl = self.clusters[new_id]

        # Subtract user's window contribution from cluster (exact within window).
        cl['S'] = cl['S'] - (S_u - np.eye(self.dim))
        cl['S_inv'] = np.linalg.inv(cl['S']).astype(np.float32, copy=False)
        cl['b'] = cl['b'] - b_u
        cl['theta_hat'] = cl['S_inv'] @ cl['b']
        cl['T'] = cl['T'] - T_i
        cl['C'].discard(user_id)

        new_cl['S'] = S_u.copy()
        new_cl['S_inv'] = np.linalg.inv(S_u).astype(np.float32, copy=False)
        new_cl['b'] = b_u.copy()
        new_cl['theta_hat'] = theta_i.copy()
        new_cl['T'] = T_i
        new_cl['C'] = {user_id}

        self.user_cluster[user_id] = new_id
        self.pivot_theta[new_id] = new_cl['theta_hat'].copy()
        self.pivot_T[new_id] = T_i
        self.arm_clust.init_cluster(new_id)
        self.n_splits += 1
        self.max_clusters_seen = max(self.max_clusters_seen, len(self.clusters))