"""
Arm clustering utility: sort items by estimated score for a user-cluster,
partition into groups, and select via two-step UCB.

MECHANICS (single-objective / scalarized):
  For each user-cluster c with estimator θ̂_c:
    1. Compute scalar scores s_a = θ̂_c^T z_a for all items.
    2. Sort items by s_a descending.
    3. Partition into n_groups equal-size groups (quantile groups).
    4. Champion of group g = the top-scoring item in that group.
  At selection:
    Step 1: Compute UCB of each champion → pick best group g*.
    Step 2: Compute UCB of all items in group g* (plus all champions for safety)
            → pick argmax.
  Cost: O((n_groups + K/n_groups) × k). Optimal n_groups = √K.

MULTI-OBJECTIVE MODE (mode='multiobj'):
  When we have D objectives and weights w, the per-cluster has D estimators
  θ̂_c^(1), ..., θ̂_c^(D). Each item has a D-dim score vector:
    s_a = (θ̂_c^(1) · z_a, ..., θ̂_c^(D) · z_a)
  The groups are obtained by K-means in R^D (with n_groups centroids).
  Champion of group g = item closest to centroid g (in D-space) with highest
  scalarized reward w^T s_a.

Our current codebase does NOT maintain per-objective cluster estimators (only the
scalarized one is tracked). So in practice the multi-objective grouping uses the
same scalarized scores as the scalarized mode. To activate TRUE multi-objective
arm clustering, one would need to maintain per-objective θ̂_c^(j) in each user-
cluster, which doubles the memory per cluster (D × k² per user-cluster).

By default we use mode='scalarized' (= current behavior). Multi-objective support
is a hook for future work — see `recluster_multiobj` below.

THREE OPT-IN BEHAVIORS (all default to False, current behavior preserved):
  - ucb_partition=True: at recluster time, sort arms by UCB index
    (mean + beta·bonus) instead of mean only. Requires V_inv and beta
    to be passed to `recluster`. Falls back to score-only otherwise.
  - update_champions_on_select=True: after each `select` call, update the
    champion of the chosen bucket to the arm with the highest UCB in that
    bucket. Reuses UCBs already computed inside `select` (cost free).
  - track_bucket_selection=True: after each `select` call, the bucket id
    of the chosen arm is written to `self.last_bucket`. -1 means "not set".

PERFORMANCE NOTE
================
`self.total_groups` is incrementally maintained whenever init_cluster /
remove_cluster / recluster modify the per-cluster champion list. This
gives base.py an O(1) lookup for the per-round metric tracking, instead
of summing over all keys of `champions` each round (which would otherwise
be linear in the number of user-clusters and ruin the per-round cost on
datasets with many users, like MovieLens with n_users ≈ 2e5).
"""
import numpy as np


class ArmClustering:
    """Manages arm quantile groups per user-cluster and two-step selection."""

    def __init__(self, n_items, n_groups, mode='scalarized',
                 ucb_partition=False,
                 update_champions_on_select=False,
                 track_bucket_selection=False):
        """
        Args:
          n_items: total number of arms.
          n_groups: target number of arm groups (e.g., √K).
          mode: 'scalarized' (default) or 'multiobj'.
          ucb_partition: if True, recluster ranks arms by UCB rather than mean.
          update_champions_on_select: if True, champion of winning bucket is
              updated after each select using already-computed UCBs.
          track_bucket_selection: if True, select() writes the winning
              bucket id to self.last_bucket.
        """
        self.n_items = n_items
        self.n_groups = max(n_groups, 1)
        self.mode = mode
        self.ucb_partition = bool(ucb_partition)
        self.update_champions_on_select = bool(update_champions_on_select)
        self.track_bucket_selection = bool(track_bucket_selection)
        # -1 means "not set yet"; written by select() when track_bucket_selection.
        self.last_bucket = -1
        self.groups = {}
        self.champions = {}
        self.top_items = {}
        # Sum of len(champions[cid]) over all cids — incrementally maintained
        # so that base.py can read it in O(1) per round (critical for
        # datasets with many users, where iterating over the dict each round
        # would dominate runtime).
        self.total_groups = 0

    def init_cluster(self, cid):
        # If cid already exists, deduct its previous contribution before
        # overwriting it; otherwise this would inflate total_groups across
        # re-initialisations.
        if cid in self.champions:
            self.total_groups -= len(self.champions[cid])
        self.groups[cid] = np.zeros(self.n_items, dtype=int)
        self.champions[cid] = [0]
        self.top_items[cid] = [np.arange(self.n_items)]
        self.total_groups += 1

    def remove_cluster(self, cid):
        if cid in self.champions:
            self.total_groups -= len(self.champions[cid])
        self.groups.pop(cid, None)
        self.champions.pop(cid, None)
        self.top_items.pop(cid, None)

    def recluster(self, cid, Z, theta_hat, V_inv=None, beta=None):
        """Partition arms into n_groups quantile buckets.

        By default, ranks by estimated mean (Z @ theta_hat). If
        self.ucb_partition is True AND V_inv and beta are provided, ranks
        by UCB (mean + beta · sqrt(z^T V^-1 z)) instead. Falls back to
        score-only if either V_inv or beta is missing.
        """
        if self.n_groups <= 1:
            self.init_cluster(cid)
            return

        if self.ucb_partition and V_inv is not None and beta is not None:
            means = Z @ theta_hat
            Vz = Z @ V_inv
            bonus_sq = np.maximum(np.einsum('ij,ij->i', Z, Vz), 0.0)
            scores = means + beta * np.sqrt(bonus_sq)
        else:
            scores = Z @ theta_hat

        sorted_idx = np.argsort(-scores)
        group_size = self.n_items // self.n_groups

        arm_group = np.zeros(self.n_items, dtype=int)
        top_list = []
        champs = []
        for g in range(self.n_groups):
            s = g * group_size
            e = (g + 1) * group_size if g < self.n_groups - 1 else self.n_items
            items = sorted_idx[s:e]
            arm_group[items] = g
            top_list.append(items)
            champs.append(int(items[0]))

        # Update the incrementally-maintained group count: subtract the
        # previous champion list length, add the new one.
        if cid in self.champions:
            self.total_groups -= len(self.champions[cid])
        self.total_groups += len(champs)

        self.groups[cid] = arm_group
        self.top_items[cid] = top_list
        self.champions[cid] = champs

    def recluster_multiobj(self, cid, Z, theta_hat_per_obj, weights, n_iter=5):
        """Multi-objective mode: K-means in D-dimensional score space.

        theta_hat_per_obj: list of D arrays of shape (k,) — one per objective.
        weights: (D,) scalarization weights for champion selection.
        """
        if self.n_groups <= 1:
            self.init_cluster(cid)
            return

        D = len(theta_hat_per_obj)
        scores = np.stack([Z @ theta_hat_per_obj[j] for j in range(D)], axis=1)
        scalar_scores = scores @ weights

        sorted_idx = np.argsort(-scalar_scores)
        init_indices = sorted_idx[::max(len(sorted_idx) // self.n_groups, 1)][:self.n_groups]
        centroids = scores[init_indices]

        for _ in range(n_iter):
            dists = np.linalg.norm(scores[:, None, :] - centroids[None, :, :], axis=2)
            arm_group = np.argmin(dists, axis=1)
            new_centroids = np.zeros_like(centroids)
            for g in range(self.n_groups):
                mask = arm_group == g
                if mask.any():
                    new_centroids[g] = scores[mask].mean(axis=0)
                else:
                    new_centroids[g] = centroids[g]
            centroids = new_centroids

        top_list = []
        champs = []
        for g in range(self.n_groups):
            items_in_g = np.where(arm_group == g)[0]
            if len(items_in_g) == 0:
                top_list.append(np.array([0]))
                champs.append(0)
                continue
            within_scores = scalar_scores[items_in_g]
            order = np.argsort(-within_scores)
            sorted_items = items_in_g[order]
            top_list.append(sorted_items)
            champs.append(int(sorted_items[0]))

        if cid in self.champions:
            self.total_groups -= len(self.champions[cid])
        self.total_groups += len(champs)

        self.groups[cid] = arm_group
        self.top_items[cid] = top_list
        self.champions[cid] = champs

    def select(self, cid, Z, theta, V_inv, beta):
        """Two-step UCB selection.

        Returns: int item_id. Side effects:
          - if self.track_bucket_selection, writes the winning bucket id to
            self.last_bucket (otherwise self.last_bucket is left untouched).
          - if self.update_champions_on_select, updates the champion of the
            winning bucket to the arm with the highest UCB among that
            bucket's own members, reusing the UCBs computed below.
        """
        champions = self.champions.get(cid, [0])
        top_items = self.top_items.get(cid, [np.arange(self.n_items)])

        if len(champions) <= 1:
            means = Z @ theta
            Vz = Z @ V_inv
            quad = np.maximum(np.einsum('ij,ij->i', Z, Vz), 0.0)
            best = int(np.argmax(means + beta * np.sqrt(quad)))
            if self.track_bucket_selection:
                self.last_bucket = 0
            return best

        champ = np.asarray(champions, dtype=np.int64)
        Zc = Z[champ]
        mc = Zc @ theta
        Vzc = Zc @ V_inv
        quad_c = np.maximum(np.einsum('ij,ij->i', Zc, Vzc), 0.0)
        best_group = int(np.argmax(mc + beta * np.sqrt(quad_c)))

        # Candidate set = items in the best group ∪ all champions.
        group_items = top_items[best_group]
        candidates = np.unique(np.concatenate((group_items, champ)))

        Zcand = Z[candidates]
        m = Zcand @ theta
        Vz = Zcand @ V_inv
        quad = np.maximum(np.einsum('ij,ij->i', Zcand, Vz), 0.0)
        ucbs = m + beta * np.sqrt(quad)
        best_local = int(np.argmax(ucbs))
        chosen_item = int(candidates[best_local])

        # Optional: refresh the winning bucket's champion using already-computed
        # UCBs. Vectorised with np.isin (no Python loop).
        if self.update_champions_on_select:
            in_group = np.isin(candidates, group_items, assume_unique=True)
            if in_group.any():
                ucbs_in_group = np.where(in_group, ucbs, -np.inf)
                champions[best_group] = int(candidates[int(np.argmax(ucbs_in_group))])

        if self.track_bucket_selection:
            self.last_bucket = best_group
        return chosen_item