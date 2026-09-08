"""
SCLUB_CA: SCLUB + Arm Clustering (no projection).
Full dim d. User clustering follows SCLUB exactly.
Arm clustering reduces per-round cost via quantile groups.

Three opt-in flags forwarded to ArmClustering (all default False):
  - ucb_partition: at recluster time, rank arms by UCB instead of mean.
  - update_champions_on_select: refresh each bucket's champion after select.
  - track_bucket_selection: log the bucket from which the chosen arm came.

Per-cluster recluster counter (vs. global counter in the original
version): a cluster is reclusterised only when it has received
arm_recluster_freq updates of its own. This avoids wasting time
reclusterising clusters that have not been touched, and matches the
per-user semantics of LinUCB_IND_RP_CA. With this change the same
"wallclock effect" as the old global counter is obtained with a
smaller arm_recluster_freq (roughly divided by the average number of
users per cluster — see freq_sweep.py).
"""
import numpy as np
from algorithms.sclub.sclub import SCLUB
from utils.arm_clustering import ArmClustering


class SCLUB_CA(SCLUB):
    def __init__(self, data, weights, lam=1.0, scale=1.0, seed=42,
                 alpha_theta=1.0, n_arm_groups=None, arm_recluster_freq=500,
                 ucb_partition=False,
                 update_champions_on_select=False,
                 track_bucket_selection=False,
                 batch=False, batch_size=20, split_merge_freq=2000):
        super().__init__(data, weights, lam, scale, seed, alpha_theta,
                         batch=batch, batch_size=batch_size,
                         split_merge_freq=split_merge_freq)
        if n_arm_groups is None:
            n_arm_groups = int(round(np.sqrt(self.n_items)))
        self.n_arm_groups = n_arm_groups
        self.arm_recluster_freq = arm_recluster_freq
        self._ucb_partition = bool(ucb_partition)
        self._update_champions = bool(update_champions_on_select)
        self._track_bucket = bool(track_bucket_selection)

    def init_run(self, T):
        super().init_run(T)
        self.arm_clust = ArmClustering(
            self.n_items, self.n_arm_groups,
            ucb_partition=self._ucb_partition,
            update_champions_on_select=self._update_champions,
            track_bucket_selection=self._track_bucket,
        )
        self.arm_clust.init_cluster(0)
        # Per-cluster counter of updates since the last arm-recluster.
        self._updates_since_recluster = {0: 0}

    def _create_cluster(self, cid):
        super()._create_cluster(cid)
        if hasattr(self, 'arm_clust'):
            self.arm_clust.init_cluster(cid)
            self._updates_since_recluster[cid] = 0

    def _sclub_beta(self, t):
        n_items = self.X.shape[0]
        return (np.sqrt(2 * np.log(2 * n_items * (t+1) / self.delta))
                + np.sqrt(self.lam))

    def recommend(self, user_id, t):
        if user_id not in self.S_u:
            self._init_user(user_id)
        self._check_phase_transition(t)
        c = self.user_cluster[user_id]
        cl = self.clusters[c]
        beta = self._sclub_beta(t)
        return self.arm_clust.select(c, self.X, cl['theta_hat'],
                                     cl['S_inv'], beta)

    def update(self, user_id, item_id, reward_vec, reward_scalar, t):
        c_before = self.user_cluster.get(user_id, 0)
        super().update(user_id, item_id, reward_vec, reward_scalar, t)
        if c_before in self._updates_since_recluster:
            self._updates_since_recluster[c_before] += 1
        else:
            self._updates_since_recluster[c_before] = 1

        # Recluster only the clusters whose per-cluster counter has
        # reached the threshold. Skipping untouched clusters keeps the
        # cost proportional to actual data flow rather than to the number
        # of clusters maintained.
        for cid in list(self._updates_since_recluster.keys()):
            if cid not in self.clusters:
                self._updates_since_recluster.pop(cid, None)
                continue
            if self._updates_since_recluster[cid] >= self.arm_recluster_freq:
                cl = self.clusters[cid]
                if self._ucb_partition:
                    beta = self._sclub_beta(t)
                    self.arm_clust.recluster(
                        cid, self.X, cl['theta_hat'],
                        V_inv=cl['S_inv'], beta=beta,
                    )
                else:
                    self.arm_clust.recluster(cid, self.X, cl['theta_hat'])
                self._updates_since_recluster[cid] = 0

    def _try_split(self, user_id, t):
        prev_n = len(self.clusters)
        super()._try_split(user_id, t)
        if len(self.clusters) > prev_n:
            new_id = max(self.clusters.keys())
            self.arm_clust.init_cluster(new_id)
            self._updates_since_recluster[new_id] = 0

    def _try_merge(self, t):
        existing_before = set(self.clusters.keys())
        super()._try_merge(t)
        existing_after = set(self.clusters.keys())
        for dead in existing_before - existing_after:
            self.arm_clust.remove_cluster(dead)
            self._updates_since_recluster.pop(dead, None)