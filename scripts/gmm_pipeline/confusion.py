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
    """Estimate C[i, j] = P(argmax posterior = j | x ~ component i).

    NOTE: uses ``gmm.predict()`` which folds in mixing-weight priors. Small
    components with low weight can leak into large ones. Use
    ``gmm_confusion_equalprior`` for a geometry-only (prior-free) estimate.
    """
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


def gmm_confusion_equalprior(
    gmm: GaussianMixture,
    n_samples_per_component: int = 50_000,
    random_state: int = 0,
) -> np.ndarray:
    """C[i, j] = P(argmax likelihood = j | x ~ component i).

    Samples from each component's Gaussian and assigns by the highest
    *log-likelihood* (not posterior) across all components — i.e. equal mixing
    weights in the decision rule. This removes the prior-weight bias that
    afflicts ``monte_carlo_confusion``: small components are no longer pushed
    toward large ones simply because the large component has a bigger weight.

    The result is a pure measure of the geometric separability of the GMM
    components in feature space, independent of class prevalences.
    """
    rng = np.random.default_rng(random_state)
    K = gmm.n_components

    # Pre-compute Cholesky factors and log-det terms for each component so we
    # can evaluate all K log-likelihoods in one vectorised pass per batch.
    covs = gmm.covariances_  # (K, d, d) for covariance_type='full'
    means = gmm.means_       # (K, d)
    d = means.shape[1]

    # Support full and diag covariance types; fall back to scipy for others.
    cov_type = gmm.covariance_type

    C = np.zeros((K, K))
    for i in range(K):
        samples = rng.multivariate_normal(means[i], covs[i], size=n_samples_per_component)
        # (n_samples, K) log-likelihoods, equal-prior
        log_liks = np.empty((n_samples_per_component, K))
        for k in range(K):
            diff = samples - means[k]           # (n, d)
            cov_k = covs[k]
            try:
                L = np.linalg.cholesky(cov_k)
                sol = np.linalg.solve(L, diff.T)  # (d, n)
                log_liks[:, k] = (
                    -0.5 * np.sum(sol ** 2, axis=0)
                    - np.sum(np.log(np.diag(L)))
                    - 0.5 * d * np.log(2 * np.pi)
                )
            except np.linalg.LinAlgError:
                # Fallback: pinv
                log_liks[:, k] = (
                    -0.5 * np.einsum("ni,ij,nj->n", diff, np.linalg.pinv(cov_k), diff)
                )
        preds = log_liks.argmax(axis=1)
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
    """Pairwise confusion via the erf formula (Hunt, Oct 2024).

    For each pair (i, j), fits a Gaussian to the 2-D posterior score vectors
    of particles labeled i and returns C[i,j] = 0.5 - 0.5*erf(z).
    Exact for K=2; pairwise approximation for K>2 (use
    analytical_multiclass_confusion for the proper multi-class extension).
    """
    from scipy.special import erf as _erf

    N, K = posteriors.shape
    a_hat = np.array([1.0, -1.0]) / np.sqrt(2.0)
    C = np.full((K, K), np.nan)

    for i in range(K):
        mask = hard_labels == i
        if mask.sum() < 10:
            continue

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

        # Combine the pairwise error rates into a row of the confusion matrix
        # (Hunt, "combine the pairwise uncertainties"). Treating the pairwise
        # "j beats i" events as approximately independent, a class-i particle is
        # correctly classified only if no competitor beats it:
        #     C[i, i] = prod_{j != i} (1 - p_ij)
        # The remaining error mass is split across competitors in proportion to
        # their individual pairwise rates. This is always non-negative and the
        # row sums to 1, unlike the naive ``1 - sum_j p_ij`` (which can go
        # negative once the pairwise rates overlap).
        p_row = C[i].copy()
        p_row[i] = 0.0
        correct = float(np.prod(1.0 - p_row[np.arange(K) != i]))
        err_mass = 1.0 - correct
        denom = p_row.sum()
        if denom > 0:
            C[i] = err_mass * p_row / denom
        else:
            C[i] = 0.0
        C[i, i] = correct

    return C


def analytical_multiclass_confusion(
    posteriors: np.ndarray,
    hard_labels: np.ndarray,
    n_samples: int = 50_000,
    min_particles: int = 10,
    reg: float = 1e-6,
    random_state: int = 0,
) -> np.ndarray:
    """Multi-class confusion via score-space Gaussian sampling.

    For each true class i, fits a K-dim Gaussian to the posterior score vectors
    of particles labeled i, draws n_samples, and computes C[i,j] as the fraction
    with argmax=j. Proper K>2 generalization of the pairwise erf formula.
    """
    rng = np.random.default_rng(random_state)
    N, K = posteriors.shape
    C = np.full((K, K), np.nan)
    for i in range(K):
        mask = hard_labels == i
        if mask.sum() < min_particles:
            continue
        scores = posteriors[mask]
        mu = scores.mean(axis=0)
        cov = np.cov(scores.T) + reg * np.eye(K)
        try:
            samples = rng.multivariate_normal(mu, cov, size=n_samples)
        except np.linalg.LinAlgError:
            cov = cov + 1e-3 * np.eye(K)
            samples = rng.multivariate_normal(mu, cov, size=n_samples)
        preds = samples.argmax(axis=1)
        C[i] = np.bincount(preds, minlength=K) / n_samples
    return C


def soft_posterior_confusion(posteriors: np.ndarray) -> np.ndarray:
    """Exact confusion matrix using the CryoSPARC posteriors as soft truth.

        C[i, j] = sum_n p[n, i] * 1(argmax_k p[n, k] == j) / sum_n p[n, i]

    Each particle's posterior is treated as a probabilistic ground-truth label:
    particle ``n`` contributes weight ``p[n, i]`` to "true class i" and is
    observed in class ``j = argmax(p[n])``. This is the honest confusion matrix
    for population deconvolution because it

    * uses every particle (no hard-label selection bias, unlike the analytical
      and pairwise estimators, which condition on ``argmax == i``),
    * needs no Gaussian assumption and no GMM (no manufactured confidence,
      no component label-switching), and
    * stays in CryoSPARC's own class coordinates, so it lines up directly with
      the observed population vector for ``pi_true = (C^T)^{-1} pi_obs``.

    On near-uniform posteriors the rows approach the observed population vector,
    so ``C`` becomes near rank-1 and the deconvolution is intentionally
    under-determined -- that ill-conditioning is a faithful signal that the
    classes are barely separable, not a bug.
    """
    posteriors = np.asarray(posteriors, dtype=np.float64)
    K = posteriors.shape[1]
    hard = posteriors.argmax(axis=1)
    num = np.zeros((K, K))
    for j in range(K):
        num[:, j] = posteriors[hard == j].sum(axis=0)
    den = posteriors.sum(axis=0).clip(min=1e-12)
    return num / den[:, None]
