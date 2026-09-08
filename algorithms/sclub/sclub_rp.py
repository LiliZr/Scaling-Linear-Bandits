"""
SCLUB_RP: SCLUB (Li et al. 2019) with Random Projection.
Same algorithm as SCLUB, but all regressions are in projected space R^k.
Working features X = Z = Φ × item_features^T.
True item_features (dim d) kept intact for reward generation (in base class).
"""
import numpy as np
from algorithms.sclub.sclub import SCLUB
from utils.utils import generate_phi_gaussian


class SCLUB_RP(SCLUB):
    def __init__(self, data, weights, k=20, lam=1.0, scale=1.0, seed=42,
                 alpha_theta=1.0, seed_proj=1,
                 batch=False, batch_size=20, split_merge_freq=2000):
        super().__init__(data, weights, lam, scale, seed, alpha_theta,
                         batch=batch, batch_size=batch_size, split_merge_freq=split_merge_freq)
        self.k = k
        self.phi = generate_phi_gaussian(self.d, k, seed=seed * seed_proj)
        self.Z = np.ascontiguousarray(self.item_features @ self.phi.T,
                                       dtype=np.float32)
        # Override: work in projected space
        self.X = self.Z
        self.dim = k
        # Pre-allocate bonus buffer at new dim (override parent's)
        self._bonus_buf = np.empty(self.n_items, dtype=np.float32)
