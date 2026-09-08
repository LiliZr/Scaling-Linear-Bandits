"""
LinUCB: a single global model in full dimension d (Li et al. 2010).
No personalization, but shared learning across all users.

Optimizations (no algorithmic change): float32 contiguous features,
pre-allocated bonus buffer, in-place rank-1 updates.
"""
import numpy as np
from algorithms.base import MultiUserLinearBandit
from utils.utils import inv_sherman_morrison


class LinUCB(MultiUserLinearBandit):

    def __init__(self, data, weights, lam=1.0, scale=1.0, seed=42,
                 batch=False, batch_size=20):
        super().__init__(data, weights, lam, scale, seed,
                         batch=batch, batch_size=batch_size)
        # Cast item features to float32 contiguous for fast BLAS.
        self.X = np.ascontiguousarray(self.item_features, dtype=np.float32)
        self._bonus_buf = np.empty(self.n_items, dtype=np.float32)

    def init_run(self, T):
        super().init_run(T)
        self.V_inv = (1.0 / self.lam) * np.eye(self.d, dtype=np.float32)
        self.b = np.zeros(self.d, dtype=np.float32)
        self.theta = np.zeros(self.d, dtype=np.float32)
        self.T_count = 0

    def recommend(self, user_id, t):
        T_count = self.T_count
        beta = (np.sqrt(2 * np.log(1/self.delta)
                        + self.dim * np.log(1 + (T_count+1)/(self.lam * self.dim)))
                + np.sqrt(self.lam))
        means = self.X @ self.theta
        Vx = self.X @ self.V_inv
        np.einsum('ij,ij->i', self.X, Vx, out=self._bonus_buf)
        np.sqrt(self._bonus_buf, out=self._bonus_buf)
        return int(np.argmax(means + beta * self._bonus_buf))

    def update(self, user_id, item_id, reward_vec, reward_scalar, t):
        x = self.X[item_id]
        np.add(self.b, reward_scalar * x, out=self.b)
        inv_sherman_morrison(self.V_inv, x)
        np.dot(self.V_inv, self.b, out=self.theta)
        self.T_count += 1



class Random(MultiUserLinearBandit):

    def __init__(self, data, weights, lam=1.0, scale=1.0, seed=42,
                 batch=False, batch_size=20):
        super().__init__(data, weights, lam, scale, seed,
                         batch=batch, batch_size=batch_size)

    def recommend(self, user_id, t):
        return self.rng.randint(self.n_items)

    def update(self, user_id, item_id, reward_vec, reward_scalar, t):
        pass