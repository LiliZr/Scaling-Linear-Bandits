"""
LinUCB-Ind: one LinUCB model per user, full dimension d.
"""
import numpy as np
from algorithms.base import MultiUserLinearBandit
from utils.utils import inv_sherman_morrison


class LinUCB_IND(MultiUserLinearBandit):

    def __init__(self, data, weights, lam=1.0, scale=1.0, seed=42,
                 batch=False, batch_size=20):
        super().__init__(data, weights, lam, scale, seed,
                         batch=batch, batch_size=batch_size)
        self.X = np.ascontiguousarray(self.item_features, dtype=np.float32)
        self._bonus_buf = np.empty(self.n_items, dtype=np.float32)

    def init_run(self, T):
        super().init_run(T)
        self.user_V_inv = {}
        self.user_b = {}
        self.user_theta = {}
        self.T_u = {}

    def _get_or_create(self, user_id):
        if user_id not in self.user_V_inv:
            self.user_V_inv[user_id] = ((1.0 / self.lam) *
                                         np.eye(self.d, dtype=np.float32))
            self.user_b[user_id] = np.zeros(self.d, dtype=np.float32)
            self.user_theta[user_id] = np.zeros(self.d, dtype=np.float32)
            self.T_u[user_id] = 0

    def recommend(self, user_id, t):
        self._get_or_create(user_id)
        T_u = self.T_u[user_id]
        beta = (np.sqrt(2 * np.log(1/self.delta)
                        + self.dim * np.log(1 + (T_u+1)/(self.lam * self.dim)))
                + np.sqrt(self.lam))
        V_inv = self.user_V_inv[user_id]
        theta = self.user_theta[user_id]
        means = self.X @ theta
        Vx = self.X @ V_inv
        np.einsum('ij,ij->i', self.X, Vx, out=self._bonus_buf)
        np.sqrt(self._bonus_buf, out=self._bonus_buf)
        return int(np.argmax(means + beta * self._bonus_buf))

    def update(self, user_id, item_id, reward_vec, reward_scalar, t):
        self._get_or_create(user_id)
        x = self.X[item_id]
        np.add(self.user_b[user_id], reward_scalar * x, out=self.user_b[user_id])
        inv_sherman_morrison(self.user_V_inv[user_id], x)
        np.dot(self.user_V_inv[user_id], self.user_b[user_id],
               out=self.user_theta[user_id])
        self.T_u[user_id] += 1
