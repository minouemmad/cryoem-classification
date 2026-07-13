"""Propagate classification uncertainty into conformational population estimates.

Model: let ``pi_true`` be the true class fractions and ``C`` the row-stochastic
confusion matrix with ``C[i, j] = P(assigned=j | true=i)``. The observed
fractions satisfy

    pi_obs = C^T @ pi_true

so a bias-corrected estimate is

    pi_true_hat = (C^T)^{-1} pi_obs

clipped to the simplex. Uncertainty bands come from a bootstrap over both the
particles and the GMM parameters.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.mixture import GaussianMixture

from .confusion import analytical_multiclass_confusion, gmm_confusion_equalprior


def observed_populations(hard_labels: np.ndarray, n_components: int) -> np.ndarray:
    counts = np.bincount(hard_labels, minlength=n_components)
    return counts / counts.sum()


def _project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection of v onto the probability simplex."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho = np.nonzero(u - cssv / (np.arange(n) + 1) > 0)[0]
    if len(rho) == 0:
        return np.full_like(v, 1.0 / n)
    rho = rho[-1]
    theta = cssv[rho] / (rho + 1)
    return np.maximum(v - theta, 0.0)


def deconvolve_populations(pi_obs: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Bias-corrected population estimate, projected onto the simplex."""
    try:
        raw = np.linalg.solve(C.T, pi_obs)
    except np.linalg.LinAlgError:
        raw, *_ = np.linalg.lstsq(C.T, pi_obs, rcond=None)
    return _project_to_simplex(raw)


def bootstrap_population_ci_analytical(
    posteriors: np.ndarray,
    hard_labels: np.ndarray,
    n_components: int,
    n_boot: int = 200,
    mc_samples: int = 50_000,
    random_state: int = 0,
    ci: float = 0.95,
):
    """Bootstrap corrected populations using the analytical confusion matrix.

    Resamples *particles only* and recomputes the analytical (score-space)
    confusion matrix on each replicate. The GMM is never refit, so there is no
    component label-switching and the error bars are consistent with the
    analytical point estimate (``deconvolve_populations(pi_obs, C_analytical)``).

    Parameters
    ----------
    posteriors : (N, K) renormalised protein-only posteriors.
    hard_labels : (N,) CryoSPARC argmax class in [0, K).
    n_components : K, number of protein classes.
    """
    rng = np.random.default_rng(random_state)
    N = len(posteriors)
    obs_boot, corr_boot = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        post_b = posteriors[idx]
        hard_b = hard_labels[idx]
        pi_obs = observed_populations(hard_b, n_components)
        C_b = analytical_multiclass_confusion(
            post_b, hard_b, n_samples=mc_samples,
            random_state=int(rng.integers(1e9)),
        )
        obs_boot.append(pi_obs)
        corr_boot.append(deconvolve_populations(pi_obs, C_b))
    obs_boot = np.array(obs_boot)
    corr_boot = np.array(corr_boot)
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return {
        "observed_mean": obs_boot.mean(0),
        "observed_std": obs_boot.std(0),
        "observed_lo": np.percentile(obs_boot, lo, axis=0),
        "observed_hi": np.percentile(obs_boot, hi, axis=0),
        "corrected_mean": corr_boot.mean(0),
        "corrected_std": corr_boot.std(0),
        "corrected_lo": np.percentile(corr_boot, lo, axis=0),
        "corrected_hi": np.percentile(corr_boot, hi, axis=0),
        "raw_observed": obs_boot,
        "raw_corrected": corr_boot,
    }


def bootstrap_gmm_parameters(
    X: np.ndarray,
    fit_fn: Callable[[np.ndarray], GaussianMixture],
    n_boot: int = 100,
    mc_samples: int = 20_000,
    random_state: int = 0,
    ci: float = 0.95,
) -> dict:
    """Bootstrap uncertainty on GMM means, covariances, weights, and confusion.

    On each replicate the GMM is refit on a bootstrap resample of *X*, then
    components are aligned to the reference fit (fitted on the full *X*) by
    the Hungarian algorithm on mean-to-mean Euclidean distances. This prevents
    label-switching across resamples from inflating the apparent variance.

    Confusion is measured via ``gmm_confusion_equalprior``: samples are drawn
    from each component's Gaussian and classified by maximum log-likelihood
    (equal-prior decision), so the result reflects purely the geometric
    separability of the GMM components — independent of class mixing weights.

    Parameters
    ----------
    X : (N, d) data fed to the GMM (after ALR/simplex transform).
    fit_fn : callable returning a fitted ``GaussianMixture`` given a resample.
        Should warm-start from the reference means to reduce label switching.
    n_boot : number of bootstrap replicates.
    mc_samples : samples per component for confusion estimation per replicate.
    ci : coverage for percentile confidence intervals (default 95 %).

    Returns
    -------
    dict with keys:
      means_mean / means_std / means_lo / means_hi : (K, d) arrays
      covs_mean  / covs_std  / covs_lo  / covs_hi  : (K, d, d) arrays (full/diag only)
      weights_mean / weights_std                   : (K,) arrays
      confusion_mean / confusion_std               : (K, K) arrays
      confusion_lo / confusion_hi                  : (K, K) percentile CIs
      raw_means / raw_covs / raw_weights / raw_confusion : all bootstrap draws
    """
    from scipy.optimize import linear_sum_assignment

    rng = np.random.default_rng(random_state)
    N = len(X)

    # Reference fit to define component ordering
    gmm_ref = fit_fn(X)
    K = gmm_ref.n_components
    cov_type = gmm_ref.covariance_type

    means_boot, covs_boot, weights_boot, confusion_boot = [], [], [], []

    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        gmm_b = fit_fn(X[idx])

        # Hungarian alignment: cost[i, j] = ||bootstrap_mean_i - ref_mean_j||
        cost = np.linalg.norm(
            gmm_b.means_[:, None, :] - gmm_ref.means_[None, :, :], axis=2
        )  # (K, K)
        _, col_ind = linear_sum_assignment(cost)
        # col_ind[i] = reference component that bootstrap component i maps to
        # order[j] = bootstrap component index corresponding to reference j
        order = np.argsort(col_ind)

        means_boot.append(gmm_b.means_[order])
        weights_boot.append(gmm_b.weights_[order])

        if cov_type in ("full", "diag"):
            covs_boot.append(gmm_b.covariances_[order])
        elif cov_type == "tied":
            covs_boot.append(gmm_b.covariances_)   # single shared matrix
        elif cov_type == "spherical":
            covs_boot.append(gmm_b.covariances_[order])

        C_b = gmm_confusion_equalprior(
            gmm_b, n_samples_per_component=mc_samples,
            random_state=int(rng.integers(1_000_000_000)),
        )
        # Reorder rows and columns to match reference component order
        C_b = C_b[np.ix_(order, order)]
        confusion_boot.append(C_b)

    means_boot = np.array(means_boot)          # (n_boot, K, d)
    weights_boot = np.array(weights_boot)      # (n_boot, K)
    confusion_boot = np.array(confusion_boot)  # (n_boot, K, K)
    covs_boot = np.array(covs_boot)

    lo_p = (1 - ci) / 2 * 100
    hi_p = (1 + ci) / 2 * 100

    result: dict = {
        "means_mean": means_boot.mean(0),
        "means_std": means_boot.std(0),
        "means_lo": np.percentile(means_boot, lo_p, axis=0),
        "means_hi": np.percentile(means_boot, hi_p, axis=0),
        "weights_mean": weights_boot.mean(0),
        "weights_std": weights_boot.std(0),
        "confusion_mean": confusion_boot.mean(0),
        "confusion_std": confusion_boot.std(0),
        "confusion_lo": np.percentile(confusion_boot, lo_p, axis=0),
        "confusion_hi": np.percentile(confusion_boot, hi_p, axis=0),
        "raw_means": means_boot,
        "raw_weights": weights_boot,
        "raw_covs": covs_boot,
        "raw_confusion": confusion_boot,
    }

    if cov_type in ("full", "diag"):
        result["covs_mean"] = covs_boot.mean(0)
        result["covs_std"] = covs_boot.std(0)
        result["covs_lo"] = np.percentile(covs_boot, lo_p, axis=0)
        result["covs_hi"] = np.percentile(covs_boot, hi_p, axis=0)

    return result
