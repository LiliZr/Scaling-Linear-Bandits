"""
Bandit_Sketch: SCLUB_RP_CA where the per-user covariance matrix S_u is
approximated by a rank-r sketch (Frequent Directions, Liberty 2013).

EVERYTHING ELSE IS IDENTICAL to SCLUB_RP_CA (and SCLUB):
  - Same split/merge tests
  - Same F(T)
  - Same phase structure
  - Same pivot snapshots
  - Same RP and arm clustering

THE ONE APPROXIMATION:
  Instead of S_u = sum of x x^T (k×k), we keep a 2r × k buffer of "compressed
  rows" produced by Frequent Directions. The implicit S_u^approx = buf^T buf,
  which has rank ≤ 2r. Memory per user: O(r × k).

  When the buffer fills up (after 2r updates without compression), we compute
  its SVD and shrink the singular values toward zero. This keeps the top-r
  directions of variance.

When a user is removed from a cluster (split), we subtract S_u^approx from the
cluster's S. This is approximate (rank-r vs full rank).
"""
import numpy as np
from algorithms.sclub.sclub_rp_ca import SCLUB_RP_CA
from utils.utils import inv_sherman_morrison


class Bandit_Sketch(SCLUB_RP_CA):
    def __init__(self, data, weights, k=20, lam=1.0, scale=1.0, seed=42,
                 alpha_theta=1.0, seed_proj=1,
                 n_arm_groups=None, arm_recluster_freq=500,
                 sketch_rank=5,
                 batch=False, batch_size=20, split_merge_freq=1):
        super().__init__(data, weights, k=k, lam=lam, scale=scale, seed=seed,
                         alpha_theta=alpha_theta, seed_proj=seed_proj,
                         n_arm_groups=n_arm_groups,
                         arm_recluster_freq=arm_recluster_freq,
                         batch=batch, batch_size=batch_size, split_merge_freq=split_merge_freq)
        self.sketch_rank = min(sketch_rank, self.k)

    def init_run(self, T):
        super().init_run(T)
        self.S_u = {}
        self.S_u_inv = {}
        self.sketch_buffer = {}  # (2r, k) per user
        self.sketch_count = {}   # current row count per user

    def _init_user(self, user_id):
        r = self.sketch_rank
        self.sketch_buffer[user_id] = np.zeros((2 * r, self.dim))
        self.sketch_count[user_id] = 0
        self.b_u[user_id] = np.zeros(self.dim)
        self.theta_u[user_id] = np.zeros(self.dim)
        self.T_u[user_id] = 0
        best = max(self.clusters.keys(), key=lambda c: len(self.clusters[c]['C']))
        self.user_cluster[user_id] = best
        self.clusters[best]['C'].add(user_id)

    def _add_to_sketch(self, u, x):
        r = self.sketch_rank
        buf = self.sketch_buffer[u]
        count = self.sketch_count[u]
        if count < 2 * r:
            buf[count] = x
            self.sketch_count[u] = count + 1
            return
        # Compress: SVD, shrink top r by sqrt(s_i^2 - s_{r+1}^2).
        _, s, Vt = np.linalg.svd(buf, full_matrices=False)
        delta_sq = s[r] ** 2 if len(s) > r else 0.0
        s_shrunk = np.sqrt(np.maximum(s[:r] ** 2 - delta_sq, 0))
        new_buf = np.zeros((2 * r, self.dim))
        for i in range(r):
            new_buf[i] = s_shrunk[i] * Vt[i]
        new_buf[r] = x
        self.sketch_buffer[u] = new_buf
        self.sketch_count[u] = r + 1

    def _S_u_approx(self, u):
        """S_u ≈ I + buf^T buf (the I is the regularization initial state)."""
        buf = self.sketch_buffer[u][:self.sketch_count[u]]
        return np.eye(self.dim) + buf.T @ buf

    def recommend(self, user_id, t):
        if user_id not in self.sketch_buffer:
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

        self._add_to_sketch(user_id, x)
        self.b_u[user_id] += reward_scalar * x
        # User estimator via sketch-based inverse: solve approximately.
        S_approx = self._S_u_approx(user_id)
        self.theta_u[user_id] = np.linalg.solve(S_approx, self.b_u[user_id])
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
        theta_i = self.theta_u[user_id]
        threshold = self.alpha_theta * (self._F(T_i) + self._F(T_pivot))
        if np.linalg.norm(theta_i - theta_pivot) <= threshold:
            return

        new_id = self.next_cluster_id
        self.next_cluster_id += 1
        self._create_cluster(new_id)
        new_cl = self.clusters[new_id]

        S_u_approx = self._S_u_approx(user_id)
        cl['S'] = cl['S'] - (S_u_approx - np.eye(self.dim))
        cl['S_inv'] = np.linalg.inv(cl['S']).astype(np.float32, copy=False)
        cl['b'] = cl['b'] - self.b_u[user_id]
        cl['theta_hat'] = cl['S_inv'] @ cl['b']
        cl['T'] = cl['T'] - T_i
        cl['C'].discard(user_id)

        new_cl['S'] = S_u_approx.copy()
        new_cl['S_inv'] = np.linalg.inv(S_u_approx).astype(np.float32, copy=False)
        new_cl['b'] = self.b_u[user_id].copy()
        new_cl['theta_hat'] = self.theta_u[user_id].copy()
        new_cl['T'] = T_i
        new_cl['C'] = {user_id}

        self.user_cluster[user_id] = new_id
        self.pivot_theta[new_id] = new_cl['theta_hat'].copy()
        self.pivot_T[new_id] = T_i
        self.arm_clust.init_cluster(new_id)
        self.n_splits += 1
        self.max_clusters_seen = max(self.max_clusters_seen, len(self.clusters))