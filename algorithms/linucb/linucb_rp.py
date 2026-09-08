"""
LinUCB-RP: a single global model in projected dimension k.
"""
import numpy as np
from algorithms.base import MultiUserLinearBandit
from utils.utils import inv_sherman_morrison, generate_phi_gaussian


class LinUCB_RP(MultiUserLinearBandit):

    def __init__(self, data, weights, k=20, lam=1.0, scale=1.0, seed=42,
                 seed_proj=1, batch=False, batch_size=20):
        super().__init__(data, weights, lam, scale, seed,
                         batch=batch, batch_size=batch_size)
        self.k = k
        self.phi = generate_phi_gaussian(self.d, k, seed=seed * seed_proj)
        self.Z = np.ascontiguousarray(self.item_features @ self.phi.T,
                                       dtype=np.float32)
        self.dim = k
        self._bonus_buf = np.empty(self.n_items, dtype=np.float32)

    def init_run(self, T):
        super().init_run(T)
        self.V_inv = (1.0 / self.lam) * np.eye(self.k, dtype=np.float32)
        self.b = np.zeros(self.k, dtype=np.float32)
        self.theta = np.zeros(self.k, dtype=np.float32)
        self.T_count = 0

    def recommend(self, user_id, t):
        T_count = self.T_count
        beta = (np.sqrt(2 * np.log(1/self.delta)
                        + self.dim * np.log(1 + (T_count+1)/(self.lam * self.dim)))
                + np.sqrt(self.lam))
        means = self.Z @ self.theta
        Vz = self.Z @ self.V_inv
        np.einsum('ij,ij->i', self.Z, Vz, out=self._bonus_buf)
        np.sqrt(self._bonus_buf, out=self._bonus_buf)
        return int(np.argmax(means + beta * self._bonus_buf))

    def update(self, user_id, item_id, reward_vec, reward_scalar, t):
        z = self.Z[item_id]
        np.add(self.b, reward_scalar * z, out=self.b)
        inv_sherman_morrison(self.V_inv, z)
        np.dot(self.V_inv, self.b, out=self.theta)
        self.T_count += 1
