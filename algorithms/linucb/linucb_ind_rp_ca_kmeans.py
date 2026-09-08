"""
LinUCB_IND_RP_CA_KMeans: per-user LinUCB in projected space, with arm
clustering AND periodic offline k-means clustering of users.

Three opt-in flags forwarded to ArmClustering (all default False):
  - ucb_partition: at recluster time, rank arms by UCB instead of mean.
  - update_champions_on_select: refresh each bucket's champion after select.
  - track_bucket_selection: log the bucket from which the chosen arm came.

A `kmeans_runs` counter is exposed so callers (and main.py) can verify how
many times the periodic k-means refresh has actually fired; this is useful
because the default kmeans_freq=2000 combined with small T or sparse user
streams sometimes never triggers k-means at all (in which case the algo
collapses to LinUCB_IND_RP_CA).
"""
import numpy as np
from algorithms.base import MultiUserLinearBandit
from utils.utils import inv_sherman_morrison, generate_phi_gaussian
from utils.arm_clustering import ArmClustering


class LinUCB_IND_RP_CA_KMeans(MultiUserLinearBandit):
    def __init__(self, data, weights, k=20, lam=1.0, scale=1.0, seed=42,
                 seed_proj=1, n_arm_groups=None, arm_recluster_freq=500,
                 n_user_clusters=30, kmeans_freq=2000,
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
        self.n_user_clusters = max(1, int(n_user_clusters))
        self.kmeans_freq = max(1, int(kmeans_freq))
        self._bonus_buf = np.empty(self.n_items, dtype=np.float32)
        self.cluster_sizes = None
        self._ucb_partition = bool(ucb_partition)
        self._update_champions = bool(update_champions_on_select)
        self._track_bucket = bool(track_bucket_selection)

    def init_run(self, T):
        super().init_run(T)
        self.user_V_inv = {}
        self.user_b = {}
        self.user_theta = {}
        self.T_u = {}
        self.user_to_cluster = {}
        self.cluster_centroids = None
        self.kmeans_runs = 0
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

    def _effective_theta(self, user_id):
        """Shrinkage: w * theta_u + (1-w) * theta_cluster_u."""
        theta_u = self.user_theta[user_id]
        if self.cluster_centroids is None or user_id not in self.user_to_cluster:
            return theta_u
        T_u = max(1, self.T_u[user_id])
        cluster_id = self.user_to_cluster[user_id]
        N_c = max(1, int(self.cluster_sizes[cluster_id]))
        w = T_u / (T_u + N_c)
        theta_c = self.cluster_centroids[cluster_id]
        return w * theta_u + (1.0 - w) * theta_c

    def recommend(self, user_id, t):
        self._get_or_create(user_id)
        T_u = self.T_u[user_id]
        beta = self._beta(T_u)
        V_inv = self.user_V_inv[user_id]
        theta_eff = self._effective_theta(user_id).astype(np.float32, copy=False)
        return self.arm_clust.select(user_id, self.Z, theta_eff, V_inv, beta)

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

        if (t + 1) % self.kmeans_freq == 0 and len(self.user_theta) >= self.n_user_clusters:
            self._run_kmeans()

    def _run_kmeans(self):
        self.kmeans_runs += 1
        user_ids = list(self.user_theta.keys())
        thetas = np.stack([self.user_theta[u] for u in user_ids]).astype(np.float32)
        n = len(user_ids)
        K = self.n_user_clusters
        rng = np.random.RandomState(self.seed + 17)
        idx = rng.choice(n, size=K, replace=(n < K))
        centroids = thetas[idx].copy()
        for _ in range(10):
            dists = np.linalg.norm(thetas[:, None, :] - centroids[None, :, :], axis=2)
            assign = np.argmin(dists, axis=1)
            new_centroids = np.zeros_like(centroids)
            counts = np.zeros(K, dtype=np.int32)
            np.add.at(new_centroids, assign, thetas)
            np.add.at(counts, assign, 1)
            for c in range(K):
                if counts[c] > 0:
                    new_centroids[c] /= counts[c]
                else:
                    new_centroids[c] = centroids[c]
            if np.allclose(centroids, new_centroids, atol=1e-5):
                break
            centroids = new_centroids
        self.cluster_centroids = centroids
        self.user_to_cluster = {u: int(assign[i]) for i, u in enumerate(user_ids)}
        sizes = np.zeros(K, dtype=np.int32)
        np.add.at(sizes, assign, 1)
        self.cluster_sizes = sizes