"""Class-repetition analysis (John Hunt's second uncertainty strategy).

We progressively add ``r`` extra components to the GMM beyond the
biologically expected ``K`` protein classes. Confused / ambiguous particles
get pulled into the duplicates, so the occupancy of each real class drops as
``r`` grows. Extrapolating occupancy vs. ``r`` toward the ``r -> infinity``
limit estimates the un-confused population.
"""
from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture


def class_repetition_analysis(
    X: np.ndarray,
    base_components: int,
    extra_range=(0, 1, 2, 3),
    init_hard=None,
    random_state: int = 0,
    covariance_type: str = "full",
    reg_covar: float = 1e-6,
    max_iter: int = 300,
    tol: float = 1e-4,
    verbose: bool = True,
):
    """Fit a sequence of GMMs with ``base_components + r`` components and
    aggregate occupancy by mapping each fitted component back to its nearest
    base-class mean."""
    base_means = (
        _means_from_hard(X, init_hard, base_components)
        if init_hard is not None else None
    )

    rng = np.random.default_rng(random_state)
    r_values, raw_w, mapped_w, mappings = [], [], [], []

    for r in extra_range:
        n = base_components + r
        if base_means is not None:
            if r > 0:
                pick = rng.integers(0, base_components, size=r)
                jitter = 0.05 * rng.standard_normal((r, X.shape[1]))
                seed = np.vstack([base_means, base_means[pick] + jitter])
            else:
                seed = base_means.copy()
        else:
            seed = None

        gmm = GaussianMixture(
            n_components=n, covariance_type=covariance_type,
            reg_covar=reg_covar, max_iter=max_iter, tol=tol,
            random_state=random_state, means_init=seed,
        ).fit(X)

        ref = base_means if base_means is not None else gmm.means_[:base_components]
        mapping = _map_to_base(gmm.means_, ref)
        agg = np.zeros(base_components)
        for k, w in enumerate(gmm.weights_):
            agg[mapping[k]] += w

        r_values.append(r)
        raw_w.append(gmm.weights_.copy())
        mapped_w.append(agg)
        mappings.append(mapping)

        if verbose:
            print(f"        r={r}: weights={np.round(gmm.weights_, 3).tolist()}  "
                  f"mapped={np.round(agg, 3).tolist()}  conv={gmm.converged_}  iters={gmm.n_iter_}")

    return {
        "r_values": r_values,
        "raw_weights": raw_w,
        "mapped_weights": np.array(mapped_w),
        "component_mapping": mappings,
    }


def _means_from_hard(X: np.ndarray, hard: np.ndarray, n: int) -> np.ndarray:
    means = np.zeros((n, X.shape[1]))
    for k in range(n):
        sel = X[hard == k]
        means[k] = sel.mean(axis=0) if len(sel) else X.mean(axis=0)
    return means


def _map_to_base(means: np.ndarray, base_means: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(means[:, None, :] - base_means[None, :, :], axis=2)
    return d.argmin(axis=1)
