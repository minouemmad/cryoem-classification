"""Embed simplex-valued posteriors in R^(K-1) so a Gaussian model is well posed.

A K-class posterior lives on the (K-1)-simplex (sums to 1, non-negative), so it
is rank-deficient. Two common embeddings:

* ``simplex_drop_last`` — drop one coordinate; cheap; preserves linearity but
  the boundary is still hard (posteriors hug 0/1).
* ``alr_transform`` — additive-log-ratio: ``log(p_i / p_ref)``; maps the open
  simplex into R^(K-1); recommended when fitting Gaussians to assignment
  probabilities because the Gaussian can extend over the full real line.
"""
from __future__ import annotations

import numpy as np


def simplex_drop_last(p: np.ndarray, drop: int = -1) -> np.ndarray:
    """Drop one column to remove the sum-to-1 redundancy."""
    return np.delete(p, drop, axis=1)


def alr_transform(p: np.ndarray, ref: int = -1, eps: float = 1e-6) -> np.ndarray:
    """Additive log-ratio transform: y_i = log(p_i / p_ref) for i != ref.

    Adds ``eps`` and renormalizes to keep zeros finite.
    """
    p = np.asarray(p, dtype=np.float64) + eps
    p = p / p.sum(axis=1, keepdims=True)
    ref_col = p[:, [ref]]
    y = np.log(np.delete(p, ref, axis=1) / ref_col)
    return y


def inverse_alr(y: np.ndarray, ref: int = -1) -> np.ndarray:
    """Inverse ALR back to the simplex (for sanity checks / visualisations)."""
    ey = np.exp(y)
    denom = 1.0 + ey.sum(axis=1, keepdims=True)
    p_rest = ey / denom
    p_ref = 1.0 / denom
    insert_at = y.shape[1] if ref == -1 else ref
    return np.insert(p_rest, insert_at, p_ref[:, 0], axis=1)
