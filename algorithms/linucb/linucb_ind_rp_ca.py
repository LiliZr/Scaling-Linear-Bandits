"""
LinUCB_IND_RP_CA: one LinUCB model per user, in projected space k,
with per-user arm clustering.

Each user has:
  - V_inv, b, theta (in R^k, projected)
  - a personal arm clustering: arms are partitioned into ~√K quantile groups
    based on this user's predicted score Z @ theta_u.

Recommendation cost per round: O(√K · k) instead of O(K · k).
This is the same speedup logic as SCLUB_CA but at the user level (since
there's no user clustering, each user IS its own "cluster" for arm grouping).

The arm groups are rebuilt every `arm_recluster_freq` rounds per user (only
when that user is observed). Between rebuilds, the existing groups are used.

Three opt-in flags forwarded to ArmClustering (all default False, current
behaviour preserved):
  - ucb_partition: at recluster time, rank arms by UCB instead of mean.
  - update_champions_on_select: refresh each bucket's champion after select.
  - track_bucket_selection: log the bucket from which the chosen arm came.
"""
import numpy as np
from algorithms.base import MultiUserLinearBandit
from utils.utils import inv_sherman_morrison, generate_phi_gaussian
from utils.arm_clustering import ArmClustering


class LinUCB_IND_RP_CA(MultiUserLinearBandit):
    def __init__(self, data, weights, k=20, lam=1.0, scale=1.0, seed=42,
                 seed_proj=1, n_arm_groups=None, arm_recluster_freq=500,
                 ucb_partition=False,
                 update_champions_on_select=False,
                 track_bucket_selection=False,
                 batch=False, batch_size=20):
        super().__init__(data, weights, lam, scale, seed,
                         batch=batch, batch_size=batch_size)
        self.k = k
        self.phi = generate_phi_gaussian(self.d, k, seed=seed * seed_proj)
        self.Z = np.ascontiguousarray(self.item_features @ self.phi.T,
                                       dtype=np.float32)
        self.dim = k
        if n_arm_groups is None:
            n_arm_groups = int(round(np.sqrt(self.n_items)))
        self.n_arm_groups = n_arm_groups
        self.arm_recluster_freq = arm_recluster_freq
        self._bonus_buf = np.empty(self.n_items, dtype=np.float32)
        self._ucb_partition = bool(ucb_partition)
        self._update_champions = bool(update_champions_on_select)
        self._track_bucket = bool(track_bucket_selection)

    def init_run(self, T):
        super().init_run(T)
        self.user_V_inv = {}
        self.user_b = {}
        self.user_theta = {}
        self.T_u = {}
        self.arm_clust = ArmClustering(
            self.n_items, self.n_arm_groups,
            ucb_partition=self._ucb_partition,
            update_champions_on_select=self._update_champions,
            track_bucket_selection=self._track_bucket,
        )

    def _beta(self, T_u):
        return (np.sqrt(2 * np.log(1/self.delta)
                        + self.dim * np.log(1 + (T_u+1)/(self.lam * self.dim)))
                + np.sqrt(self.lam))

    def _get_or_create(self, user_id):
        if user_id not in self.user_V_inv:
            self.user_V_inv[user_id] = ((1.0 / self.lam) *
                                         np.eye(self.k, dtype=np.float32))
            self.user_b[user_id] = np.zeros(self.k, dtype=np.float32)
            self.user_theta[user_id] = np.zeros(self.k, dtype=np.float32)
            self.T_u[user_id] = 0
            self.arm_clust.init_cluster(user_id)
            self.arm_clust.recluster(user_id, self.Z, self.user_theta[user_id])

    def recommend(self, user_id, t):
        self._get_or_create(user_id)
        T_u = self.T_u[user_id]
        beta = self._beta(T_u)
        V_inv = self.user_V_inv[user_id]
        theta = self.user_theta[user_id]
        return self.arm_clust.select(user_id, self.Z, theta, V_inv, beta)

    def update(self, user_id, item_id, reward_vec, reward_scalar, t):
        self._get_or_create(user_id)
        z = self.Z[item_id]
        np.add(self.user_b[user_id], reward_scalar * z, out=self.user_b[user_id])
        inv_sherman_morrison(self.user_V_inv[user_id], z)
        np.dot(self.user_V_inv[user_id], self.user_b[user_id],
               out=self.user_theta[user_id])
        self.T_u[user_id] += 1

        if self.T_u[user_id] > 0 and self.T_u[user_id] % self.arm_recluster_freq == 0:
            if self._ucb_partition:
                beta = self._beta(self.T_u[user_id])
                self.arm_clust.recluster(
                    user_id, self.Z, self.user_theta[user_id],
                    V_inv=self.user_V_inv[user_id], beta=beta,
                )
            else:
                self.arm_clust.recluster(
                    user_id, self.Z, self.user_theta[user_id],
                )