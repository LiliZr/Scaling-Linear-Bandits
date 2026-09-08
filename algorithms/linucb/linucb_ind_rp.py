"""
LinUCB-Ind-RP: one LinUCB model per user in projected space k.
Fully personalized, projected. No info sharing between users.
"""
import numpy as np
from algorithms.base import MultiUserLinearBandit
from utils.utils import inv_sherman_morrison, generate_phi_gaussian


class LinUCB_IND_RP(MultiUserLinearBandit):
    def __init__(self, data, weights, k=20, lam=1.0, scale=1.0, seed=42, seed_proj=1, batch=False, batch_size=20):
        super().__init__(data, weights, lam, scale, seed, batch=batch, batch_size=batch_size)
        self.k = k
        self.phi = generate_phi_gaussian(self.d, k, seed=seed * seed_proj)
        # Cast Z to float32 for faster BLAS. Also store contiguous.
        self.Z = np.ascontiguousarray(self.item_features @ self.phi.T,
                                       dtype=np.float32)
        self.dim = k
        # Pre-allocate the bonus buffer (used every round).
        self._bonus_buf = np.empty(self.n_items, dtype=np.float32)
        # Pre-cache beta multiplier components that don't depend on T_u
        # (the log term is recomputed on demand).
        self._beta_const = (np.sqrt(2 * np.log(1/self.delta)) + np.sqrt(self.lam))

    def init_run(self, T):
        super().init_run(T)
        self.user_V_inv = {}
        self.user_b = {}
        self.user_theta = {}
        self.T_u = {}

    def _get_or_create(self, user_id):
        if user_id not in self.user_V_inv:
            self.user_V_inv[user_id] = ((1.0 / self.lam) *
                                         np.eye(self.k, dtype=np.float32))
            self.user_b[user_id] = np.zeros(self.k, dtype=np.float32)
            self.user_theta[user_id] = np.zeros(self.k, dtype=np.float32)
            self.T_u[user_id] = 0


    def recommend(self, user_id, t):
        self._get_or_create(user_id)
        T_u = self.T_u[user_id]
        # beta = const + extra log term. The first term is precomputed.
        beta = (np.sqrt(2 * np.log(1/self.delta)
                        + self.dim * np.log(1 + (T_u+1)/(self.lam * self.dim)))
                + np.sqrt(self.lam))
        V_inv = self.user_V_inv[user_id]
        theta = self.user_theta[user_id]
        # means: (K,). Z @ theta with contiguous float32 is BLAS-fast.
        means = self.Z @ theta
        # bonuses: sqrt(diag(Z V_inv Z^T)). The einsum form is the fastest
        # numpy-pure approach for this without batched gemm.
        Vz = self.Z @ V_inv
        # Use the precomputed bonus_buf to avoid an allocation per round.
        np.einsum('ij,ij->i', self.Z, Vz, out=self._bonus_buf)
        np.sqrt(self._bonus_buf, out=self._bonus_buf)
        return int(np.argmax(means + beta * self._bonus_buf))

    def update(self, user_id, item_id, reward_vec, reward_scalar, t):
        self._get_or_create(user_id)
        z = self.Z[item_id]
        np.add(self.user_b[user_id], reward_scalar * z, out=self.user_b[user_id])
        inv_sherman_morrison(self.user_V_inv[user_id], z)
        np.dot(self.user_V_inv[user_id], self.user_b[user_id],
               out=self.user_theta[user_id])
        self.T_u[user_id] += 1

