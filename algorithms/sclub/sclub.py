"""
SCLUB (Li, Chen, Li, Leung — IJCAI 2019).
"Improved Algorithm on Online Clustering of Bandits".

Faithful implementation following the paper's Algorithms 1-4.
Full dimension d, no projection, no arm clustering.

Key formulas from the paper:
- Per-user profile: (S_i, b_i, T_i) where S_i = sum of x x^T, b_i = sum of y x
- Per-cluster profile: S^j = I + Σ_{i∈C^j} (S_i - I), b^j = Σ_{i∈C^j} b_i
- Split (user i_τ from cluster j): ||θ̂_i - θ̃^j|| > α_θ (F(T_i) + F(T̃^j))
  where F(T) = sqrt((1 + ln(1+T)) / (1+T))
- Merge (clusters j1, j2): ||θ̂^{j1} - θ̂^{j2}|| < α_θ/2 * (F(T^{j1}) + F(T^{j2}))
- Phases: s-th phase contains 2^s rounds; checks are done in phase structure.

The "working features" X are stored in self.X — subclasses can override
(e.g., SCLUB_RP sets self.X = self.Z, the projected features).
self.item_features stays intact — used by the base class for reward generation.
"""
import numpy as np
from algorithms.base import MultiUserLinearBandit
from utils.utils import inv_sherman_morrison


class SCLUB(MultiUserLinearBandit):
    def __init__(self, data, weights, lam=1.0, scale=1.0, seed=42,
                 alpha_theta=1.0, split_merge_freq=2000,
                 batch=False, batch_size=20):
        """
        SCLUB (Li et al. 2019, IJCAI). Faithful implementation: split/merge
        tests run at EVERY round, as per Algorithm 1. The phase loop `s`
        controls only the pivot recomputation and the user_checked reset.

        Args:
            split_merge_freq: NON-PAPER. Default 1 = paper exactly. Higher
                values run the tests only every `split_merge_freq` rounds,
                trading off cluster reactivity for speed. The cluster
                detection latency grows by at most (freq-1) rounds.
        """
        super().__init__(data, weights, lam, scale, seed,
                         batch=batch, batch_size=batch_size)
        self.alpha_theta = alpha_theta
        self.split_merge_freq = max(1, int(split_merge_freq))
        self.X = np.ascontiguousarray(self.item_features, dtype=np.float32)
        self.dim = self.X.shape[1]
        self._bonus_buf = np.empty(self.n_items, dtype=np.float32)

    def init_run(self, T):
        super().init_run(T)
        self.S_u = {}
        self.S_u_inv = {}
        self.b_u = {}
        self.theta_u = {}
        self.T_u = {}
        self.user_cluster = {}
        self.clusters = {}
        self._create_cluster(0)
        self.next_cluster_id = 1
        self.pivot_theta = {0: np.zeros(self.dim)}
        self.pivot_T = {0: 0}
        self.phase = 1
        self.phase_start = 0
        self.phase_length = 2 ** self.phase
        self.user_checked = set()
        # Incrementally maintained set of "checked clusters" (clusters where
        # every user has been observed in the current phase). This avoids
        # re-scanning all clusters at every round in _try_merge.
        # SCLUB paper, Algorithm 4: merge only operates on checked clusters.
        self.checked_cids = set()
        self.n_splits = 0
        self.n_merges = 0
        self.max_clusters_seen = 1

    def _create_cluster(self, cid):
        self.clusters[cid] = {
            'S': np.eye(self.dim, dtype=np.float32),
            'S_inv': np.eye(self.dim, dtype=np.float32),
            'b': np.zeros(self.dim, dtype=np.float32),
            'theta_hat': np.zeros(self.dim, dtype=np.float32),
            'T': 0,
            'C': set(),
        }

    def _F(self, T):
        return np.sqrt((1.0 + np.log(1.0 + T)) / (1.0 + T))

    def _init_user(self, user_id):
        self.S_u[user_id] = np.eye(self.dim, dtype=np.float32)
        self.S_u_inv[user_id] = np.eye(self.dim, dtype=np.float32)
        self.b_u[user_id] = np.zeros(self.dim, dtype=np.float32)
        self.theta_u[user_id] = np.zeros(self.dim, dtype=np.float32)
        self.T_u[user_id] = 0
        best = max(self.clusters.keys(), key=lambda c: len(self.clusters[c]['C']))
        self.user_cluster[user_id] = best
        self.clusters[best]['C'].add(user_id)

    def _check_phase_transition(self, t):
        if t - self.phase_start >= self.phase_length:
            self.phase += 1
            self.phase_start = t
            self.phase_length = 2 ** self.phase
            self.user_checked.clear()
            self.checked_cids.clear()
            for cid, cl in self.clusters.items():
                self.pivot_theta[cid] = cl['theta_hat'].copy()
                self.pivot_T[cid] = cl['T']

    def recommend(self, user_id, t):
        if user_id not in self.S_u:
            self._init_user(user_id)
        self._check_phase_transition(t)
        c = self.user_cluster[user_id]
        cl = self.clusters[c]
        n_items = self.X.shape[0]
        beta = (np.sqrt(2 * np.log(2 * n_items * (t+1) / self.delta)) + np.sqrt(self.lam))
        means = self.X @ cl['theta_hat']
        Sx = self.X @ cl['S_inv']
        np.einsum('ij,ij->i', self.X, Sx, out=self._bonus_buf)
        np.sqrt(self._bonus_buf, out=self._bonus_buf)
        return int(np.argmax(means + beta * self._bonus_buf))

    def update(self, user_id, item_id, reward_vec, reward_scalar, t):
        c = self.user_cluster[user_id]
        cl = self.clusters[c]
        x = self.X[item_id]
        # Single rank-1 outer product, reused for both user and cluster S.
        xx = np.multiply.outer(x, x)

        self.S_u[user_id] += xx
        inv_sherman_morrison(self.S_u_inv[user_id], x)
        np.add(self.b_u[user_id], reward_scalar * x, out=self.b_u[user_id])
        np.dot(self.S_u_inv[user_id], self.b_u[user_id],
               out=self.theta_u[user_id])
        self.T_u[user_id] += 1

        cl['S'] += xx
        inv_sherman_morrison(cl['S_inv'], x)
        np.add(cl['b'], reward_scalar * x, out=cl['b'])
        np.dot(cl['S_inv'], cl['b'], out=cl['theta_hat'])
        cl['T'] += 1

        # SCLUB (Li et al. 2019, Algorithm 1, lines 13-15): split and merge
        # tests are run AT EVERY ROUND. The phase structure (loop `s`) only
        # controls when the pivot θ̃_j is recomputed and when user_checked is
        # reset — NOT when the tests run.
        #
        # IMPLEMENTATION NOTE (paper-faithful): the paper's Algorithm 4 only
        # operates on CHECKED clusters (clusters where all users have been
        # observed in the current phase). At round τ we just observed user
        # `user_id`, so the only cluster whose checked-status could change
        # this round is the user's own cluster. We track this incrementally
        # in `self.checked_cids` to avoid scanning all clusters at every
        # round. Merge tests only fire when `user_id`'s cluster has *just*
        # become checked (i.e., its last unchecked user has been observed).
        if (t + 1) % self.split_merge_freq == 0:
            source_cluster = self.user_cluster[user_id]
            self._try_split(user_id, t)
            self.user_checked.add(user_id)
            # Two clusters may have just become "checked" by this round:
            #   1. The user's CURRENT cluster (post-split, could be new_id or same)
            #   2. If split happened, the SOURCE cluster also lost an unchecked user
            #      and its remaining users might now all be checked.
            for c_check in {source_cluster, self.user_cluster[user_id]}:
                if c_check not in self.clusters:
                    continue  # cluster was merged away in a previous step
                if c_check in self.checked_cids:
                    continue
                cluster_users = self.clusters[c_check]['C']
                if cluster_users and cluster_users.issubset(self.user_checked):
                    self.checked_cids.add(c_check)
            # Merge only when at least 2 clusters are checked.
            if len(self.checked_cids) >= 2:
                self._try_merge(t)
        else:
            self.user_checked.add(user_id)

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
        diff = np.linalg.norm(theta_i - theta_pivot)
        if diff <= threshold:
            return

        new_id = self.next_cluster_id
        self.next_cluster_id += 1
        self._create_cluster(new_id)

        cl['S'] = cl['S'] - self.S_u[user_id] + np.eye(self.dim, dtype=np.float32)
        cl['S_inv'] = np.linalg.inv(cl['S']).astype(np.float32, copy=False)
        cl['b'] = cl['b'] - self.b_u[user_id]
        cl['theta_hat'] = (cl['S_inv'] @ cl['b']).astype(np.float32, copy=False)
        cl['T'] = cl['T'] - T_i
        cl['C'].discard(user_id)

        new_cl = self.clusters[new_id]
        new_cl['S'] = self.S_u[user_id].copy()
        new_cl['S_inv'] = self.S_u_inv[user_id].copy()
        new_cl['b'] = self.b_u[user_id].copy()
        new_cl['theta_hat'] = self.theta_u[user_id].copy()
        new_cl['T'] = T_i
        new_cl['C'] = {user_id}

        self.user_cluster[user_id] = new_id
        self.pivot_theta[new_id] = new_cl['theta_hat'].copy()
        self.pivot_T[new_id] = T_i

        self.n_splits += 1
        self.max_clusters_seen = max(self.max_clusters_seen, len(self.clusters))

    def _try_merge(self, t):
        # Use the incrementally-maintained set of checked clusters.
        # This is exactly the set the paper's Algorithm 4 iterates over.
        checked_cids = [cid for cid in self.checked_cids if cid in self.clusters]
        if len(checked_cids) < 2:
            return
        merged_into = {}  # cid -> destination cid (after merging)
        for i, c1 in enumerate(checked_cids):
            if c1 in merged_into or c1 not in self.clusters:
                continue
            for c2 in checked_cids[i+1:]:
                if c2 in merged_into or c2 not in self.clusters:
                    continue
                cl1 = self.clusters[c1]
                cl2 = self.clusters[c2]
                threshold = (self.alpha_theta / 2.0) * (self._F(cl1['T']) + self._F(cl2['T']))
                diff = np.linalg.norm(cl1['theta_hat'] - cl2['theta_hat'])
                if diff < threshold:
                    cl1['S'] = cl1['S'] + cl2['S'] - np.eye(self.dim, dtype=np.float32)
                    cl1['S_inv'] = np.linalg.inv(cl1['S']).astype(np.float32, copy=False)
                    cl1['b'] = cl1['b'] + cl2['b']
                    cl1['theta_hat'] = (cl1['S_inv'] @ cl1['b']).astype(np.float32, copy=False)
                    cl1['T'] += cl2['T']
                    cl1['C'] |= cl2['C']
                    for u in cl2['C']:
                        self.user_cluster[u] = c1
                    del self.clusters[c2]
                    self.checked_cids.discard(c2)
                    self.pivot_theta.pop(c2, None)
                    self.pivot_T.pop(c2, None)
                    merged_into[c2] = c1
                    self.n_merges += 1