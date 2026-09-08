import numpy as np
import math
from scipy.linalg import hadamard


def inv_sherman_morrison(B, u):
    """
    Efficient inverse update: B := (B^{-1} + uu^T)^{-1}.
    Modifies B in-place.

    Faster than the textbook version by avoiding the reshape+transpose chain
    and unnecessary intermediate allocations. Uses:
        Bu = B @ u            (k,)
        gamma = 1 + u @ Bu    scalar
        B -= outer(Bu, Bu) / gamma
    """
    Bu = B @ u
    gamma = 1.0 + float(u @ Bu)
    # In-place rank-1 subtraction.
    B -= np.multiply.outer(Bu, Bu) / gamma


def generate_phi(d, n, k=None, c=0.001, eps=0.1, p=2, c1=1, seed=None):
    """
    Generate FJLT projection matrix Φ ∈ R^{k×d}
    """
    rng = np.random.RandomState(seed)
    if k is None:
        k = math.ceil(c * eps**(-2) * np.log10(n))
    
    power = np.log2(d)
    next_dim_power_2 = d if power == int(power) else 2**(int(power)+1)

    q = min(1, (c1 * (eps**(p-2)) * (np.log10(n)**p)) / next_dim_power_2)
    P = np.zeros((k, next_dim_power_2))
    probabilities = rng.random((k, next_dim_power_2))
    n_samples = len(probabilities[probabilities <= q])
    P[probabilities <= q] = rng.normal(0, np.sqrt(1 / q), size=n_samples)

    H = hadamard(next_dim_power_2)[:, :d] * (1. / np.sqrt(next_dim_power_2))
    diag = rng.choice([-1, 1], p=[0.5, 0.5], size=d)

    return (1 / np.sqrt(k)) * P @ (H * diag)


def generate_phi_gaussian(d, k, seed=None):
    """
    Simple Gaussian projection matrix Φ ∈ R^{k×d}
    Each entry ~ N(0, 1/k)
    """
    rng = np.random.RandomState(seed)
    return rng.normal(0, 1.0 / np.sqrt(k), size=(k, d))
