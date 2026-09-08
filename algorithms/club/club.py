"""
CLUB (Gentile, Li, Zappella — ICML 2014)
"Online Clustering of Bandits"

Faithful implementation following Figure 1 of the paper.

PROTOCOL (paper notation):
Initialization: G_0 = complete graph on n users, M_{i,0} = I, b_{i,0} = 0 for all i.
At each round t = 1, 2, ...:
  1. Receive user i_t.
  2. Compute clusters: connected components of G_{t-1}. Let C_t be the cluster of i_t.
     Aggregate: bar_M = I + Σ_{i ∈ C_t} (M_i - I), bar_b = Σ_{i ∈ C_t} b_i.
     Cluster prototype: bar_w = bar_M^{-1} bar_b.
  3. Select arm: k_t = argmax_{x ∈ A_t} [bar_w^T x + α * ||x||_{bar_M^{-1}} * sqrt(log(1+t))]
     (we use a standard LinUCB bonus on the aggregated matrix).
  4. Observe reward y_t.
  5. Update served user:
       M_{i_t} = M_{i_t} + x_t x_t^T
       b_{i_t} = b_{i_t} + y_t x_t
       hat_w_{i_t} = M_{i_t}^{-1} b_{i_t}
       T_{i_t}  = T_{i_t} + 1
  6. Graph update: for each neighbor j of i_t in G_{t-1}:
       If ||hat_w_{i_t} - hat_w_j|| > CB(T_{i_t}) + CB(T_j):
           delete edge (i_t, j)
     where CB(T) = alpha_2 * sqrt((1 + log(1 + T)) / (1 + T))
  7. G_t = updated graph.

Note: two α parameters in the paper (α for UCB exploration, α_2 for edge deletion).
We expose both: `scale` (UCB bonus) and `alpha_cb` (edge deletion).
"""
import numpy as np
from algorithms.base import MultiUserLinearBandit
from utils.utils import inv_sherman_morrison


class CLUB(MultiUserLinearBandit):
    def __init__(self, data, weights, lam=1.0, scale=1.0, seed=42,
                 alpha_cb=2.0, batch=False, batch_size=20):
        super().__init__(data, weights, lam, scale, seed,
                         batch=batch, batch_size=batch_size)
        self.alpha_cb = alpha_cb
        # CLUB rebuilds M_inv via np.linalg.inv (float64), so keep float64 here
        # for type consistency. The other gains (cached optimal, sampled memory,
        # in-place ops) still apply.
        self.X = np.ascontiguousarray(self.item_features, dtype=np.float64)
        self.dim = self.X.shape[1]
        self._bonus_buf = np.empty(self.n_items, dtype=np.float64)

    def init_run(self, T):
        super().init_run(T)

        self.n_users = len(self.user_thetas)

        # Per-user state
        self.M = {}          # M_i = I + Σ x x^T (ridge: lambda=1 implicit)
        self.M_inv = {}
        self.b = {}
        self.w_hat = {}      # user estimator
        self.T_i = {}        # observation count

        # Initialize all users; in the paper, the graph is over [n] from the start
        for u in range(self.n_users):
            self._init_user(u)

        # Graph G_0 = complete graph, stored as adjacency sets
        # For efficiency with n large, we store edges by pair (undirected).
        self.adj = {u: set() for u in range(self.n_users)}
        for u in range(self.n_users):
            for v in range(u + 1, self.n_users):
                self.adj[u].add(v)
                self.adj[v].add(u)

        # Connected components as clusters: initially one big cluster
        # user_comp[u] = component_id. Maintained lazily (recomputed when edges deleted).
        self.user_comp = {u: 0 for u in range(self.n_users)}
        self.components = {0: set(range(self.n_users))}
        self.next_comp_id = 1
        # Aggregated cluster stats (cached, invalidated on edge deletion)
        self.cluster_cache = {}  # comp_id → {M_inv, w_hat}
        self._build_cache(0)

        self.n_splits = 0
        self.n_merges = 0  # CLUB does not merge; kept for API compatibility
        self.max_clusters_seen = 1

    def _init_user(self, u):
        self.M[u] = np.eye(self.dim)
        self.M_inv[u] = np.eye(self.dim)
        self.b[u] = np.zeros(self.dim)
        self.w_hat[u] = np.zeros(self.dim)
        self.T_i[u] = 0

    def _CB(self, T):
        """α_2 * sqrt((1 + log(1 + T)) / (1 + T)) — edge deletion confidence bound."""
        return self.alpha_cb * np.sqrt((1.0 + np.log(1.0 + T)) / (1.0 + T))

    def _build_cache(self, comp_id):
        """Compute aggregate M, b, and cache M_inv + w_hat for the component."""
        members = self.components[comp_id]
        I = np.eye(self.dim)
        M_agg = I.copy()
        b_agg = np.zeros(self.dim)
        for u in members:
            M_agg += self.M[u] - I
            b_agg += self.b[u]
        M_inv = np.linalg.inv(M_agg)
        self.cluster_cache[comp_id] = {
            'M_inv': M_inv,
            'w_hat': M_inv @ b_agg,
        }

    def recommend(self, user_id, t):
        c = self.user_comp[user_id]
        cache = self.cluster_cache[c]
        w = cache['w_hat']
        M_inv = cache['M_inv']
        beta = np.sqrt(2 * np.log(t + 2) + self.dim * np.log(1 + (t+1)/self.dim))
        means = self.X @ w
        Mx = self.X @ M_inv
        np.einsum('ij,ij->i', self.X, Mx, out=self._bonus_buf)
        np.sqrt(self._bonus_buf, out=self._bonus_buf)
        return int(np.argmax(means + beta * self._bonus_buf))

    def update(self, user_id, item_id, reward_vec, reward_scalar, t):
        x = self.X[item_id]
        xx = np.outer(x, x)

        # Update user i_t
        self.M[user_id] += xx
        inv_sherman_morrison(self.M_inv[user_id], x)
        self.b[user_id] += reward_scalar * x
        self.w_hat[user_id] = self.M_inv[user_id] @ self.b[user_id]
        self.T_i[user_id] += 1

        # Edge deletion: check edges from user_id to neighbors
        to_delete = []
        cb_i = self._CB(self.T_i[user_id])
        for j in self.adj[user_id]:
            if self.T_i[j] < 1:
                continue  # no data for j yet; can't compare
            cb_j = self._CB(self.T_i[j])
            diff = np.linalg.norm(self.w_hat[user_id] - self.w_hat[j])
            if diff > cb_i + cb_j:
                to_delete.append(j)

        if to_delete:
            for j in to_delete:
                self.adj[user_id].discard(j)
                self.adj[j].discard(user_id)
            # Recompute connected component containing user_id
            self._recompute_component(user_id)

        # Always rebuild aggregate cache for user's current component
        # (the component changed: members' M_i and b_i were updated)
        c = self.user_comp[user_id]
        self._build_cache(c)

    def _recompute_component(self, u):
        """BFS from u; if u's component has changed (split), re-partition."""
        old_comp_id = self.user_comp[u]
        old_members = self.components[old_comp_id]

        # BFS from u
        visited = {u}
        queue = [u]
        while queue:
            v = queue.pop()
            for w in self.adj[v]:
                if w not in visited:
                    visited.add(w)
                    queue.append(w)

        if visited == old_members:
            return  # no split

        # Split: find all disjoint components within old_members
        remaining = old_members - visited
        comps = [visited]
        while remaining:
            seed_u = next(iter(remaining))
            comp = {seed_u}
            queue = [seed_u]
            while queue:
                v = queue.pop()
                for w in self.adj[v]:
                    if w not in comp and w in old_members:
                        comp.add(w)
                        queue.append(w)
            comps.append(comp)
            remaining -= comp

        # Invalidate old
        del self.components[old_comp_id]
        self.cluster_cache.pop(old_comp_id, None)

        # First comp keeps old_comp_id, others get new ids
        self.components[old_comp_id] = comps[0]
        for u_ in comps[0]:
            self.user_comp[u_] = old_comp_id
        self._build_cache(old_comp_id)

        for comp in comps[1:]:
            cid = self.next_comp_id
            self.next_comp_id += 1
            self.components[cid] = comp
            for u_ in comp:
                self.user_comp[u_] = cid
            self._build_cache(cid)
            self.n_splits += 1

        self.max_clusters_seen = max(self.max_clusters_seen, len(self.components))

    # Expose `clusters` alias for uniform reporting across algorithms
    @property
    def clusters(self):
        return self.components
