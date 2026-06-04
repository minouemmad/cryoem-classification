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

from .confusion import monte_carlo_confusion, analytical_multiclass_confusion


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


def bootstrap_population_ci(
    X: np.ndarray,
    fit_fn: Callable[[np.ndarray], GaussianMixture],
    n_boot: int = 50,
    mc_samples: int = 20_000,
    random_state: int = 0,
    ci: float = 0.95,
):
    """Bootstrap (particles + refit) the corrected populations.

    Parameters
    ----------
    X : (N, d) data fed to the GMM.
    fit_fn : callable returning a fitted ``GaussianMixture`` given a resample.
    Returns dict with mean, std, lower, upper for both observed and corrected
    populations.
    """
    rng = np.random.default_rng(random_state)
    N = len(X)
    obs_boot, corr_boot = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        gmm_b = fit_fn(X[idx])
        hard_b = gmm_b.predict(X[idx])
        pi_obs = observed_populations(hard_b, gmm_b.n_components)
        C_b = monte_carlo_confusion(gmm_b, mc_samples, random_state=int(rng.integers(1e9)))
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

    Unlike ``bootstrap_population_ci`` this resamples *particles only* and
    recomputes the analytical (score-space) confusion matrix on each replicate.
    The GMM is never refit, so there is no component label-switching and the
    error bars are consistent with the analytical point estimate
    (``deconvolve_populations(pi_obs, C_analytical)``).

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
