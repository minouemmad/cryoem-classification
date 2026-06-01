"""Compute pairwise misclassification probabilities from a fitted GMM.

The class-confusion matrix entry ``C[i, j]`` is

    P(assigned = j | true = i)

where ``assigned`` means ``argmax_k pi_k N(x | mu_k, Sigma_k)`` under the
fitted mixture. The matrix is estimated by Monte Carlo sampling from each
fitted component and re-classifying the samples through the full mixture
(Hunt notes, ``HSHuntScoreClassificationProblem10252024a``). A closed-form
Bhattacharyya-distance bound is also provided as a sanity check.
"""
from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture


def monte_carlo_confusion(
    gmm: GaussianMixture,
    n_samples_per_component: int = 50_000,
    random_state: int = 0,
) -> np.ndarray:
    """Estimate C[i, j] = P(argmax posterior = j | x ~ component i)."""
    rng = np.random.default_rng(random_state)
    K = gmm.n_components
    C = np.zeros((K, K))
    for i in range(K):
        mean = gmm.means_[i]
        cov = gmm.covariances_[i]
        samples = rng.multivariate_normal(mean, cov, size=n_samples_per_component)
        preds = gmm.predict(samples)
        counts = np.bincount(preds, minlength=K)
        C[i] = counts / counts.sum()
    return C


def bhattacharyya_distance(m1, S1, m2, S2) -> float:
    """Bhattacharyya distance between two multivariate Gaussians."""
    S = 0.5 * (S1 + S2)
    d = m1 - m2
    sign, logdet_S = np.linalg.slogdet(S)
    _, logdet_1 = np.linalg.slogdet(S1)
    _, logdet_2 = np.linalg.slogdet(S2)
    return 0.125 * d @ np.linalg.solve(S, d) + 0.5 * (logdet_S - 0.5 * (logdet_1 + logdet_2))


def bhattacharyya_pairwise(gmm: GaussianMixture) -> np.ndarray:
    """Pairwise Bhattacharyya distance matrix (symmetric, zero diagonal).

    Upper bound on Bayes-error overlap: ``BC = exp(-D_B)``; the closer to 1,
    the more confusable the two components are.
    """
    K = gmm.n_components
    D = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            D[i, j] = D[j, i] = bhattacharyya_distance(
                gmm.means_[i], gmm.covariances_[i],
                gmm.means_[j], gmm.covariances_[j],
            )
    return D


def hard_assignment_confusion(
    original_hard: np.ndarray,
    gmm_hard: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Empirical agreement matrix between CryoSPARC and GMM hard assignments
    (rows = CryoSPARC class, cols = GMM component, row-normalised)."""
    M = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        sel = gmm_hard[original_hard == i]
        if len(sel) == 0:
            continue
        counts = np.bincount(sel, minlength=n_classes)
        M[i] = counts / counts.sum()
    return M


def analytical_pairwise_confusion(
    posteriors: np.ndarray,
    hard_labels: np.ndarray,
) -> np.ndarray:
    """Exact pairwise confusion via the erf formula (Hunt classification notes).

    For each ordered pair (i, j) with i ≠ j, the method

    1. Selects particles hard-assigned to class i.
    2. Extracts their 2-D posterior score vector (S_i, S_j) from ``posteriors``.
    3. Fits a Gaussian: empirical mean m and covariance C_cov.
    4. Projects onto the difference axis a = [1, -1] / sqrt(2):
           z  = (m · a) / sqrt(2 * a^T C_cov a)
       which measures how far the mean score-difference is from the decision
       boundary in units of the projected standard deviation.
    5. Returns the misclassification probability
           C[i, j] = 0.5 - 0.5 * erf(z)

    This formula is *exact* for the 2-class Gaussian classifier.  For K > 2
    classes it is a *pairwise approximation* — each (i,j) pair is evaluated
    independently, ignoring competing classes.  The diagonal is set to
    1 − sum_of_off-diagonal so each row sums to 1.

    Reference: Hunt, H.S., "Score Classification Problem", Oct 2024.

    Parameters
    ----------
    posteriors : (N, K) array
        Raw class-posterior probabilities from CryoSPARC (not ALR-transformed).
    hard_labels : (N,) int array
        Hard class assignment for each particle (0-indexed, length K classes).

    Returns
    -------
    C : (K, K) float array
        Row-normalised pairwise confusion matrix.  NaN rows indicate classes
        with fewer than 10 particles.
    """
    from scipy.special import erf as _erf

    N, K = posteriors.shape
    a_hat = np.array([1.0, -1.0]) / np.sqrt(2.0)
    C = np.full((K, K), np.nan)

    for i in range(K):
        mask = hard_labels == i
        if mask.sum() < 10:
            continue

        off_sum = 0.0
        for j in range(K):
            if i == j:
                continue

            # 2-D score vectors for true-class-i particles
            scores = posteriors[np.ix_(mask, [i, j])]   # (N_i, 2)
            m = scores.mean(axis=0)                      # mean [S_i, S_j]
            cov = np.cov(scores.T)                       # 2×2 sample covariance

            m_dot_a = float(a_hat @ m)                   # (m_i − m_j) / sqrt(2)
            var_proj = float(a_hat @ cov @ a_hat)        # projected variance

            if var_proj <= 0 or not np.isfinite(var_proj):
                C[i, j] = 0.0
            else:
                z = m_dot_a / np.sqrt(2.0 * var_proj)
                C[i, j] = 0.5 - 0.5 * float(_erf(z))

            off_sum += C[i, j]

        C[i, i] = max(0.0, 1.0 - off_sum)

    return C
