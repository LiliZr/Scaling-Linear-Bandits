import numpy as np
import time
import tracemalloc


class MultiUserLinearBandit:
    """
    Base class for multi-user linear bandit algorithms.
    Handles the main loop, reward generation, and metric tracking.

    OPTIMIZATIONS vs naive base (do not affect algorithm logic or output):
      - Per-user optimal rewards CACHED at init: computed once per user,
        reused at each round instead of recomputing all (K, D) matvecs.
      - Memory tracking SAMPLED (every 100 rounds + last round) instead of
        every round, since tracemalloc.get_traced_memory() is slow.
      - User_stream cast to int32 contiguous array for faster indexing.
      - Optional batch mode: when batch=True, observations are buffered for
        `batch_size` rounds, then updates are applied in a single pass.
        Recommendations within the batch use the model AS OF the previous
        flush (stale model). This changes the algorithm's online behaviour
        and is opt-in only.
    """
    def __init__(self, data, weights, lam=1.0, scale=1.0, seed=42,
                 batch=False, batch_size=20):
        self.rng = np.random.RandomState(seed)
        self.data = data
        self.weights = weights
        self.lam = lam
        self.scale = scale
        self.seed = seed

        self.n_items = data['n_items']
        self.n_users = data['n_users']
        self.d = data['d']
        self.D_obj = data['D_objectives']
        self.noise_std = data['noise_std']
        self.item_features = data['item_features']
        self.user_thetas = data['user_thetas']

        self.delta = 0.01
        self.dim = self.item_features.shape[1]

        # Batch settings
        self.batch = batch
        self.batch_size = max(1, int(batch_size))

        # ─── Pre-compute per-user optimal scalar reward ───
        # This is a (constant) function of the user only and is used for the
        # regret computation every round. Computing it once saves O(K·d·D)
        # flops per round (~800K flops for K=4K, d=100, D=2).
        self._optimal_per_user = None  # populated in init_run

    def init_run(self, T):
        self.cumulative_reward = np.zeros(T)
        self.cumulative_regret = np.zeros(T)
        self.cpu_time = np.zeros(T)
        self.wall_time = np.zeros(T)
        self.memory_peak = np.zeros(T)
        self.n_user_clusters_over_time = np.zeros(T, dtype=np.int32)
        self.n_arm_groups_over_time = np.zeros(T, dtype=np.int32)
        # Bucket id from which each selected arm was drawn. -1 = not tracked
        # (no arm clustering, or track_bucket_selection=False on the
        # ArmClustering instance). Filled per-round in the main loop.
        self.bucket_over_time = np.full(T, -1, dtype=np.int32)

        # Pre-compute the optimal scalar reward per user (used in regret).
        # Shape: (n_users,). Cached once; used by compute_optimal_reward fast path.
        # user_thetas can be either a dict {u: (D, d)} or a 3-D array (n_users, D, d).
        # In MovieLens-loaded data it is a (n_users, D, d) ndarray.
        # Pre-compute the optimal scalar reward per user (used in regret).
        # Shape: (n_users,). Cached once; used by compute_optimal_reward fast path.
        # user_thetas can be either a dict {u: (D, d)} or a 3-D array (n_users, D, d).
        # In MovieLens-loaded data it is a (n_users, D, d) ndarray.
        #
        # We compute the optimum by BATCHES of users instead of all at once,
        # because the naive einsum 'kd,uDd,D->uk' on a large dataset
        # (e.g. ML-32M with n_users=200k, n_items=87k) would allocate a
        # n_users × n_items intermediate array (~140 GB), which crashes or
        # swaps to disk. Processing in user-batches keeps the peak memory
        # bounded by `batch × n_items × 4 bytes` (e.g. 1.7 GB for batch=5000).
        # The result is mathematically identical to the previous global einsum.
        # This entire block runs in init_run, BEFORE t0_cpu = process_time(),
        # so it does not enter the cpu_time measurement.
        ut = self.user_thetas
        if isinstance(ut, np.ndarray) and ut.ndim == 3:
            self._optimal_per_user = np.empty(self.n_users, dtype=np.float32)
            # Pre-scalarise thetas once: (n_users, D, d) · (D,) → (n_users, d)
            scal_thetas = np.einsum('uDd,D->ud', ut, self.weights)
            X = self.item_features  # (n_items, d)
            BATCH = 5000  # memory peak ≈ BATCH × n_items × 4 bytes
            for start in range(0, self.n_users, BATCH):
                end = min(start + BATCH, self.n_users)
                # scores[u_in_batch, k] = <scal_thetas[u], X[k]>
                scores = X @ scal_thetas[start:end].T  # (n_items, batch)
                self._optimal_per_user[start:end] = scores.max(axis=0)
        else:
            # Dict-like: iterate (slower but only at init).
            self._optimal_per_user = np.zeros(self.n_users, dtype=np.float32)
            for u in range(self.n_users):
                theta_u = ut[u] if (u in ut if hasattr(ut, '__contains__') else True) else np.zeros((self.D_obj, self.d))
                all_rewards = self.item_features @ theta_u.T
                scalar = all_rewards @ self.weights
                self._optimal_per_user[u] = scalar.max()

    def compute_optimal_reward(self, user_id):
        """Best possible scalar reward for this user. O(1) thanks to caching."""
        return self._optimal_per_user[user_id]

    def generate_reward(self, user_id, item_id):
        theta_u = self.user_thetas[user_id]
        x_a = self.item_features[item_id]
        reward_vec = theta_u @ x_a + self.rng.randn(self.D_obj) * self.noise_std
        reward_scalar = self.weights @ reward_vec
        return reward_vec, reward_scalar

    def recommend(self, user_id, t):
        raise NotImplementedError

    def update(self, user_id, item_id, reward_vec, reward_scalar, t):
        raise NotImplementedError

    def _track_state(self, t):
        """Track cluster counts at round t. Cheap (O(1) per round).

        For algos that hold a user-cluster dict `self.clusters`
        (SCLUB-family), the user-cluster count is len(self.clusters).
        For algos that use periodic k-means user clustering
        (`self.user_to_cluster`), it is the number of distinct centroids.
        Otherwise it is 1 (a single implicit "global" cluster).

        For arm groups, we rely on the incrementally-maintained
        `arm_clust.total_groups` attribute when present, which is O(1) per
        round. This is critical on datasets with many users (e.g.
        MovieLens, n_users ≈ 2e5): previously this method summed
        `len(champions[cid])` over every cid each round, which scaled
        linearly with the number of observed users and dominated the
        per-round cost. The behaviour is identical for users; only the
        cost is different.
        """
        # --- user clusters ---
        if hasattr(self, 'clusters'):
            try:
                self.n_user_clusters_over_time[t] = len(self.clusters)
            except TypeError:
                self.n_user_clusters_over_time[t] = 0
        elif getattr(self, 'user_to_cluster', None):
            self.n_user_clusters_over_time[t] = len(
                set(self.user_to_cluster.values()))
        else:
            self.n_user_clusters_over_time[t] = 1

        # --- arm groups (O(1) via the incrementally maintained counter) ---
        if hasattr(self, 'arm_clust'):
            tg = getattr(self.arm_clust, 'total_groups', None)
            if tg is not None:
                self.n_arm_groups_over_time[t] = int(tg)
            else:
                # Defensive fallback (older ArmClustering without total_groups);
                # only executed once per round, still acceptable but slower.
                self.n_arm_groups_over_time[t] = sum(
                    len(v) for v in self.arm_clust.champions.values())
        else:
            self.n_arm_groups_over_time[t] = 0

    def run(self, user_stream, T=None):
        if T is None:
            T = len(user_stream)
        T = min(T, len(user_stream))

        # Make sure user_stream is a fast NumPy int array (avoid Python int).
        user_stream = np.asarray(user_stream, dtype=np.int64)

        tracemalloc.start()
        self.init_run(T)
        t0_cpu = time.process_time()
        t0_wall = time.perf_counter()

        cum_reward = 0.0
        cum_regret = 0.0

        # Memory sampling interval (we only call get_traced_memory every
        # `mem_sample_every` rounds + at the end). The remaining slots are
        # filled by forward-fill.
        mem_sample_every = max(1, T // 100)  # ~100 samples over the run
        last_mem_mb = 0.0

        if self.batch:
            self._run_batched(user_stream, T, t0_cpu, t0_wall,
                              cum_reward, cum_regret,
                              mem_sample_every)
        else:
            self._run_online(user_stream, T, t0_cpu, t0_wall,
                             cum_reward, cum_regret,
                             mem_sample_every)

        tracemalloc.stop()
        return self.cumulative_regret

    def _run_online(self, user_stream, T, t0_cpu, t0_wall,
                    cum_reward, cum_regret, mem_sample_every):
        """Standard online loop: recommend → observe → update per round."""
        last_mem_mb = 0.0
        # Cache the tracking flag once: if False, the per-round check
        # below skips a useless attribute lookup.
        track_bucket = (
            hasattr(self, 'arm_clust')
            and getattr(self.arm_clust, 'track_bucket_selection', False)
        )
        for t in range(T):
            user_id = int(user_stream[t])
            item_id = self.recommend(user_id, t)
            if track_bucket:
                self.bucket_over_time[t] = int(self.arm_clust.last_bucket)
            reward_vec, reward_scalar = self.generate_reward(user_id, item_id)
            optimal_reward = self._optimal_per_user[user_id]
            self.update(user_id, item_id, reward_vec, reward_scalar, t)

            cum_reward += reward_scalar
            cum_regret += (optimal_reward - reward_scalar)
            self.cumulative_reward[t] = cum_reward
            self.cumulative_regret[t] = cum_regret
            self.cpu_time[t] = time.process_time() - t0_cpu
            self.wall_time[t] = time.perf_counter() - t0_wall

            if t % mem_sample_every == 0 or t == T - 1:
                current, _ = tracemalloc.get_traced_memory()
                last_mem_mb = current / (1024 * 1024)
            self.memory_peak[t] = last_mem_mb

            self._track_state(t)

    def _run_batched(self, user_stream, T, t0_cpu, t0_wall,
                     cum_reward, cum_regret, mem_sample_every):
        """Batched loop: buffer B observations, then apply all updates at once.

        The recommendation policy WITHIN a batch uses the model as of the
        previous flush (the model is "stale" for ~B rounds). At the end of
        each batch of size B, all updates are applied in sequence.
        """
        B = self.batch_size
        last_mem_mb = 0.0
        buffer = []
        track_bucket = (
            hasattr(self, 'arm_clust')
            and getattr(self.arm_clust, 'track_bucket_selection', False)
        )

        for t in range(T):
            user_id = int(user_stream[t])
            item_id = self.recommend(user_id, t)
            if track_bucket:
                self.bucket_over_time[t] = int(self.arm_clust.last_bucket)
            reward_vec, reward_scalar = self.generate_reward(user_id, item_id)
            optimal_reward = self._optimal_per_user[user_id]
            buffer.append((t, user_id, item_id, reward_vec, reward_scalar,
                           optimal_reward))

            cum_reward += reward_scalar
            cum_regret += (optimal_reward - reward_scalar)
            self.cumulative_reward[t] = cum_reward
            self.cumulative_regret[t] = cum_regret

            # Flush at end of batch or last round.
            if len(buffer) >= B or t == T - 1:
                for (tb, uid, iid, rv, rs, opt) in buffer:
                    self.update(uid, iid, rv, rs, tb)
                buffer.clear()

            self.cpu_time[t] = time.process_time() - t0_cpu
            self.wall_time[t] = time.perf_counter() - t0_wall

            if t % mem_sample_every == 0 or t == T - 1:
                current, _ = tracemalloc.get_traced_memory()
                last_mem_mb = current / (1024 * 1024)
            self.memory_peak[t] = last_mem_mb

            self._track_state(t)