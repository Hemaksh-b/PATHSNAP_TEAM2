"""Emission log-probabilities for the HMM (Issue #7).

* ``classifier_emission_logp`` — primary: log of Member 2's candidate probability.
* ``gaussian_emission_logp``  — fallback / ablation: GPS noise modelled as a
  zero-mean Gaussian on the perpendicular point-to-arc distance (metres).
"""
from __future__ import annotations

import numpy as np

DEFAULT_EPS = 1e-9


def classifier_emission_logp(prob: np.ndarray, eps: float = DEFAULT_EPS) -> np.ndarray:
    """log(p) with p clipped to [eps, 1] so zero probabilities stay finite."""
    p = np.asarray(prob, dtype=float)
    if np.isnan(p).any():
        raise ValueError("probabilities contain NaN")
    return np.log(np.clip(p, eps, 1.0))


def gaussian_emission_logp(dist_m: np.ndarray, sigma_m: float) -> np.ndarray:
    """log N(dist; 0, sigma^2) for perpendicular GPS-to-arc distances in metres."""
    if sigma_m <= 0:
        raise ValueError("sigma_m must be positive")
    d = np.asarray(dist_m, dtype=float)
    return -0.5 * (d / sigma_m) ** 2 - np.log(sigma_m * np.sqrt(2.0 * np.pi))
