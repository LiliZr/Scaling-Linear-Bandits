"""
LinUCB_RP_CA: a single global LinUCB model in projected dimension k,
augmented with arm clustering for fast arm selection.

There is a SINGLE arm partition (shared by all users) since there is
a single global theta. The partition is rebuilt every `arm_recluster_freq`
rounds.

Three opt-in flags (all default False, current behavior preserved):
  - ucb_partition: at recluster time, rank arms by UCB instead of mean.
  - update_champions_on_select: refresh each bucket's champion after select.
  - track_bucket_selection: log the bucket from which the chosen arm came.
"""
import numpy as np
from algorithms.base import MultiUserLinearBandit
from utils.utils import inv_sherman_morrison, generate_phi_gaussian
from utils.arm_clustering import ArmClustering


_GLOBAL_KEY = 0


class LinUCB_RP_CA(MultiUserLinearBandit):
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
        self.V_inv = (1.0 / self.lam) * np.eye(self.k, dtype=np.float32)
        self.b = np.zeros(self.k, dtype=np.float32)
        self.theta = np.zeros(self.k, dtype=np.float32)
        self.T_count = 0
        self.arm_clust = ArmClustering(
            self.n_items, self.n_arm_groups,
            ucb_partition=self._ucb_partition,
            update_champions_on_select=self._update_champions,
            track_bucket_selection=self._track_bucket,
        )
        self.arm_clust.init_cluster(_GLOBAL_KEY)
        self.arm_clust.recluster(_GLOBAL_KEY, self.Z, self.theta)

    def _beta(self, T_count):
        return (np.sqrt(2 * np.log(1/self.delta)
                        + self.dim * np.log(1 + (T_count+1)/(self.lam * self.dim)))
                + np.sqrt(self.lam))

    def recommend(self, user_id, t):
        beta = self._beta(self.T_count)
        return self.arm_clust.select(_GLOBAL_KEY, self.Z, self.theta,
                                     self.V_inv, beta)

    def update(self, user_id, item_id, reward_vec, reward_scalar, t):
        z = self.Z[item_id]
        np.add(self.b, reward_scalar * z, out=self.b)
        inv_sherman_morrison(self.V_inv, z)
        np.dot(self.V_inv, self.b, out=self.theta)
        self.T_count += 1

        if self.T_count > 0 and self.T_count % self.arm_recluster_freq == 0:
            if self._ucb_partition:
                beta = self._beta(self.T_count)
                self.arm_clust.recluster(
                    _GLOBAL_KEY, self.Z, self.theta,
                    V_inv=self.V_inv, beta=beta,
                )
            else:
                self.arm_clust.recluster(_GLOBAL_KEY, self.Z, self.theta)