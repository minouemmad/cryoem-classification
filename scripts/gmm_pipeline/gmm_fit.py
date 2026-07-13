"""Fit a Gaussian Mixture Model to (transformed) posterior vectors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.mixture import GaussianMixture


@dataclass
class GMMResult:
    model: GaussianMixture
    X: np.ndarray                  # data fitted on (N, d)
    responsibilities: np.ndarray   # (N, n_components)
    hard_labels: np.ndarray        # (N,) argmax of responsibilities
    converged: bool
    bic: float
    aic: float
    log_likelihood: float


def _init_means_from_hard(X: np.ndarray, hard: np.ndarray, n: int) -> np.ndarray:
    means = np.zeros((n, X.shape[1]))
    for k in range(n):
        sel = X[hard == k]
        means[k] = sel.mean(axis=0) if len(sel) else X.mean(axis=0)
    return means


def fit_gmm(
    X: np.ndarray,
    n_components: int,
    init_hard: Optional[np.ndarray] = None,
    covariance_type: str = "full",
    reg_covar: float = 1e-6,
    max_iter: int = 500,
    tol: float = 1e-5,
    random_state: int = 0,
    n_init: int = 1,
) -> GMMResult:
    """Fit a GMM. If ``init_hard`` is given, means are seeded from per-class
    centroids (warm start from the existing CryoSPARC hard assignment)."""
    kwargs = dict(
        n_components=n_components,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        n_init=n_init,
    )
    if init_hard is not None:
        means_init = _init_means_from_hard(X, init_hard, n_components)
        kwargs["means_init"] = means_init

    gmm = GaussianMixture(**kwargs).fit(X)
    resp = gmm.predict_proba(X)
    return GMMResult(
        model=gmm,
        X=X,
        responsibilities=resp,
        hard_labels=resp.argmax(axis=1),
        converged=bool(gmm.converged_),
        bic=float(gmm.bic(X)),
        aic=float(gmm.aic(X)),
        log_likelihood=float(gmm.score(X) * len(X)),
    )


def gmm_diagnostics(result: GMMResult) -> dict:
    """Per-component sanity stats: weight, mean norm, condition number."""
    m = result.model
    diag = {
        "converged": result.converged,
        "n_iter": int(m.n_iter_),
        "bic": result.bic,
        "aic": result.aic,
        "weights": m.weights_.tolist(),
        "component_cond_number": [
            float(np.linalg.cond(c)) for c in m.covariances_
        ],
    }
    return diag
