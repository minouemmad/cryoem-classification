"""Lightweight diagnostic plots for the GMM uncertainty pipeline."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse, Patch
from matplotlib.lines import Line2D
import numpy as np


def plot_confusion(C: np.ndarray, labels=None, title="Confusion matrix", out=None):
    fig, ax = plt.subplots(figsize=(4 + 0.4 * len(C), 3.5 + 0.4 * len(C)))
    im = ax.imshow(C, vmin=0, vmax=1, cmap="viridis")
    K = len(C)
    labels = labels or [f"C{i}" for i in range(K)]
    ax.set_xticks(range(K), labels)
    ax.set_yticks(range(K), labels)
    ax.set_xlabel("Assigned")
    ax.set_ylabel("True")
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center",
                    color="white" if C[i,j] < 0.5 else "black", fontsize=9)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return fig


def plot_population_ci(observed, corrected, lo_or_std, hi=None, labels=None, out=None):
    """Bar chart of observed vs corrected populations.

    Pass ``lo_or_std`` and ``hi`` as CI bounds, OR pass only ``lo_or_std`` as
    a symmetric ±std array (``hi`` omitted or None) to draw ±std error bars.
    """
    K = len(observed)
    x = np.arange(K)
    labels = labels or [f"C{i}" for i in range(K)]
    fig, ax = plt.subplots(figsize=(1.4 * K + 2, 4))
    ax.bar(x - 0.18, observed, width=0.35, label="Observed (CryoSPARC)", color="#888")
    ax.bar(x + 0.18, corrected, width=0.35, label="Corrected (soft-posterior)", color="#3a7")
    if hi is not None:
        yerr_lo = np.maximum(0, corrected - lo_or_std)
        yerr_hi = np.maximum(0, hi - corrected)
        ax.errorbar(x + 0.18, corrected,
                    yerr=[yerr_lo, yerr_hi],
                    fmt="none", ecolor="black", capsize=4)
    else:
        ax.errorbar(x + 0.18, corrected,
                    yerr=lo_or_std,
                    fmt="none", ecolor="black", capsize=4)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Population fraction")
    ax.set_title("Conformational populations: observed vs bias-corrected")
    ax.legend()
    fig.tight_layout()
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return fig


def plot_population_comparison(observed, corrections: dict, labels=None, out=None):
    """Grouped bar chart of the corrected population from every confusion matrix.

    ``corrections`` is an ordered ``{matrix_name: (K,) fractions}`` mapping.
    Observed populations are drawn first (grey), then one bar per matrix so the
    effect of each inversion step is directly comparable per class.
    """
    K = len(observed)
    names = list(corrections.keys())
    n_series = len(names) + 1
    x = np.arange(K)
    labels = labels or [f"C{i}" for i in range(K)]
    width = 0.8 / n_series

    fig, ax = plt.subplots(figsize=(1.7 * K + 3, 4.5))
    offset = -0.4 + width / 2
    ax.bar(x + offset, observed, width=width, label="Observed (CryoSPARC)",
           color="#888888", edgecolor="white", linewidth=0.4)
    palette = plt.get_cmap("tab10").colors
    for k, name in enumerate(names):
        offset = -0.4 + width / 2 + (k + 1) * width
        ax.bar(x + offset, corrections[name], width=width, label=name,
               color=palette[k % len(palette)], edgecolor="white", linewidth=0.4)

    ax.set_xticks(x, labels)
    ax.set_ylabel("Population fraction")
    ax.set_title("Corrected populations at each matrix-inversion step")
    ax.legend(fontsize=8, ncol=2, framealpha=0.85)
    ax.margins(y=0.12)
    fig.tight_layout()
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return fig


def plot_repetition(r_values, mapped_weights, labels=None, out=None):
    K = mapped_weights.shape[1]
    labels = labels or [f"C{i}" for i in range(K)]
    fig, ax = plt.subplots(figsize=(6, 4))
    for k in range(K):
        ax.plot(r_values, mapped_weights[:, k], "o-", label=labels[k])
    ax.set_xlabel("# extra (duplicate) components r")
    ax.set_ylabel("Aggregated weight of base class")
    ax.set_title("Class-repetition analysis")
    ax.legend()
    fig.tight_layout()
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Helpers for GMM landscape / NLL plots
# ---------------------------------------------------------------------------

_COLORS = plt.get_cmap("tab10").colors


def _get_full_cov(gmm, k: int) -> np.ndarray:
    """Return a (D, D) covariance array for component k regardless of cov type."""
    D = gmm.means_.shape[1]
    ct = gmm.covariance_type
    if ct == "full":
        return gmm.covariances_[k]
    if ct == "tied":
        return gmm.covariances_
    if ct == "diag":
        return np.diag(gmm.covariances_[k])
    if ct == "spherical":
        return np.eye(D) * gmm.covariances_[k]
    raise ValueError(f"Unknown covariance_type: {ct}")


def _pca_project_gmm(X: np.ndarray, gmm):
    """Project X and GMM parameters into the top-2 PCA directions of X.

    Returns
    -------
    X_2d         : (N, 2) projected data
    means_2d     : (K, 2) projected component means
    covs_2d      : (K, 2, 2) projected component covariances
    ev_ratio     : (2,) explained-variance ratios
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    means_2d = pca.transform(gmm.means_)
    V = pca.components_  # (2, D)
    covs_2d = np.array([V @ _get_full_cov(gmm, k) @ V.T for k in range(gmm.n_components)])
    return X_2d, means_2d, covs_2d, pca.explained_variance_ratio_


def _draw_confidence_ellipses(ax, mean, cov, n_std_list, color, base_alpha=0.22):
    """Draw filled confidence ellipses at each level in n_std_list (largest first)."""
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)
    angle = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
    for i, ns in enumerate(sorted(n_std_list, reverse=True)):
        w, h = 2.0 * ns * np.sqrt(vals)
        face_rgba = mcolors.to_rgba(color, alpha=base_alpha / ns)
        edge_rgba = mcolors.to_rgba(color, alpha=min(base_alpha / ns * 4, 0.65))
        ell = Ellipse(
            xy=mean, width=w, height=h, angle=angle,
            facecolor=face_rgba, edgecolor=edge_rgba,
            linewidth=0.9, zorder=2,
        )
        ell.set_clip_box(ax.bbox)
        ax.add_patch(ell)


# ---------------------------------------------------------------------------
# Combined GMM landscape: NLL surface + equi-probability ellipses + scatter
# ---------------------------------------------------------------------------

def plot_gmm_landscape(
    gmm, X: np.ndarray, hard_labels: np.ndarray,
    labels=None, title="Negative log-likelihood predicted by GMM",
    out=None, n_std=(1, 2, 3),
    scatter_alpha=0.3, scatter_size=3, grid_n=260,
):
    """NLL contourf background + data scatter + equi-probability ellipses.

    For 2-D ALR data (K=3) axes are raw ALR coordinates.
    For higher-D data (K>3) everything is projected onto the top-2 PCA directions.
    """
    from scipy.stats import multivariate_normal as _mvn

    K = gmm.n_components
    labels = labels or [f"C{i}" for i in range(K)]

    if X.shape[1] == 2:
        X_2d = X
        means_2d = gmm.means_
        covs_2d = np.array([_get_full_cov(gmm, k) for k in range(K)])
        xlabel, ylabel, note = "ALR dim 1", "ALR dim 2", None
    else:
        X_2d, means_2d, covs_2d, ev = _pca_project_gmm(X, gmm)
        xlabel = f"PC1  ({ev[0]:.1%} var)"
        ylabel = f"PC2  ({ev[1]:.1%} var)"
        note = f"PCA 2-D projection — {ev[0] + ev[1]:.1%} of total variance shown"

    # --- NLL evaluation grid ---
    pad = 0.12
    x0, x1 = X_2d[:, 0].min(), X_2d[:, 0].max()
    y0, y1 = X_2d[:, 1].min(), X_2d[:, 1].max()
    gx = np.linspace(x0 - (x1 - x0) * pad, x1 + (x1 - x0) * pad, grid_n)
    gy = np.linspace(y0 - (y1 - y0) * pad, y1 + (y1 - y0) * pad, grid_n)
    xx, yy = np.meshgrid(gx, gy)
    pts = np.column_stack([xx.ravel(), yy.ravel()])

    log_p = np.full(len(pts), -np.inf)
    for k in range(K):
        lp = _mvn.logpdf(pts, mean=means_2d[k], cov=covs_2d[k], allow_singular=True)
        log_p = np.logaddexp(log_p, np.log(gmm.weights_[k] + 1e-300) + lp)
    nll = (-log_p).reshape(xx.shape)

    finite = nll[np.isfinite(nll)]
    vmin = np.percentile(finite, 1)
    vmax = np.percentile(finite, 97)
    nll_disp = np.clip(nll, vmin, vmax)

    fig, ax = plt.subplots(figsize=(7, 6))
    cf = ax.contourf(xx, yy, nll_disp, levels=50, cmap="viridis_r", alpha=0.88)
    ax.contour(xx, yy, nll_disp, levels=10, colors="white", linewidths=0.3, alpha=0.25)
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("−log p(x)  [lower = higher density]", fontsize=9)

    # --- scatter (rasterised) ---
    for k in range(K):
        mask = hard_labels == k
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            s=scatter_size, alpha=scatter_alpha,
            color=_COLORS[k % len(_COLORS)],
            rasterized=True,
        )

    # --- confidence ellipses ---
    for k in range(K):
        _draw_confidence_ellipses(ax, means_2d[k], covs_2d[k], n_std,
                                  color=_COLORS[k % len(_COLORS)])

    # --- centre markers: colored X with white outline ---
    for k in range(K):
        ax.plot(*means_2d[k], "x", color=_COLORS[k % len(_COLORS)],
                ms=11, mew=3, zorder=10, markeredgecolor=_COLORS[k % len(_COLORS)],
                path_effects=[__import__('matplotlib.patheffects', fromlist=['withStroke']).withStroke(linewidth=5, foreground='white')])

    # --- legend: colored Patch per class; sigma levels as text annotation ---
    legend_handles = [
        Patch(facecolor=_COLORS[k % len(_COLORS)], alpha=0.75,
              label=f"{labels[k]}  (w={gmm.weights_[k]:.3f})")
        for k in range(K)
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.8)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    if note:
        ax.text(0.01, 0.01, note, transform=ax.transAxes,
                fontsize=7, color="white", va="bottom")
    fig.tight_layout()
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Class table: per-class corrected populations + accuracy from every matrix
# ---------------------------------------------------------------------------

def plot_class_table(
    labels,
    pi_obs,
    corrections: dict,
    accuracies: dict,
    primary: str,
    pi_corr_std=None,
    pi_corr_lo=None,
    pi_corr_hi=None,
    extra_metrics: dict | None = None,
    title: str = "",
    out=None,
):
    """One table comparing every confusion matrix side by side.

    For each class (row) the table shows the observed fraction, then — for each
    confusion matrix — the deconvolved (corrected) population and that matrix's
    diagonal accuracy ``C[i, i]``. The ``primary`` matrix additionally gets a
    95% bootstrap CI column and its correction column is highlighted.

    Parameters
    ----------
    labels        : class labels.
    pi_obs        : (K,) observed (CryoSPARC-hard) population fractions.
    corrections   : ordered ``{matrix_name: (K,) corrected fractions}``; the
                    primary matrix should be listed first.
    accuracies    : ordered ``{matrix_name: (K,) diagonal accuracy}``; same keys
                    as ``corrections``.
    primary       : key in ``corrections`` that carries the bootstrap CI.
    pi_corr_std / pi_corr_lo / pi_corr_hi : optional CI arrays for ``primary``.
    extra_metrics : optional ``{column_label: (K,) values in [0, 1]}`` appended
                    on the right (e.g. max Bhattacharyya overlap).
    """
    K = len(labels)
    names = list(corrections.keys())
    extra_metrics = extra_metrics or {}
    has_ci = pi_corr_lo is not None and pi_corr_hi is not None

    col_headers = ["Class", "Obs."]
    if has_ci:
        col_headers.append(f"{primary}\n95% CI")
    for name in names:
        tag = " *" if name == primary else ""
        col_headers.append(f"{name}{tag}\nCorr")
        col_headers.append(f"{name}\nAcc")
    col_headers.extend(extra_metrics.keys())

    rows = []
    for i, lbl in enumerate(labels):
        row = [lbl, f"{pi_obs[i]:.1%}"]
        if has_ci:
            row.append(f"{pi_corr_lo[i]:.0%}–{pi_corr_hi[i]:.0%}")
        for name in names:
            corr = corrections[name][i]
            if name == primary and pi_corr_std is not None:
                row.append(f"{corr:.1%}\n±{pi_corr_std[i]:.1%}")
            else:
                row.append(f"{corr:.1%}")
            row.append(f"{accuracies[name][i]:.1%}")
        for vals in extra_metrics.values():
            row.append(f"{vals[i]:.2f}")
        rows.append(row)

    ncols = len(col_headers)
    fig, ax = plt.subplots(figsize=(1.15 * ncols + 1.5, 0.42 * (K + 1) + 0.8))
    ax.axis("off")

    tbl = ax.table(cellText=rows, colLabels=col_headers,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(list(range(ncols)))
    tbl.scale(1, 2.0)

    # column index bookkeeping
    base = 3 if has_ci else 2
    acc_cols, corr_cols, primary_corr_col = {}, {}, None
    for k, name in enumerate(names):
        corr_cols[name] = base + 2 * k
        acc_cols[name] = base + 2 * k + 1
        if name == primary:
            primary_corr_col = base + 2 * k

    header_bg = "#d9d9d9"
    for j in range(ncols):
        cell = tbl[(0, j)]
        cell.set_facecolor(header_bg)
        cell.set_text_props(color="black", fontweight="bold")
        cell.set_edgecolor("#aaaaaa")

    acc_cmap = plt.get_cmap("RdYlGn")
    acc_col_set = set(acc_cols.values())
    for i in range(K):
        bg = "white" if i % 2 == 0 else "#f5f5f5"
        for j in range(ncols):
            cell = tbl[(i + 1, j)]
            cell.set_facecolor(bg)
            cell.set_edgecolor("#cccccc")
        # accuracy columns: RdYlGn heat
        for name, col in acc_cols.items():
            val = float(accuracies[name][i])
            tbl[(i + 1, col)].set_facecolor(acc_cmap(val))
            tbl[(i + 1, col)].set_text_props(
                color="white" if val < 0.45 or val > 0.85 else "black")
        # primary correction column: subtle highlight
        if primary_corr_col is not None:
            tbl[(i + 1, primary_corr_col)].set_facecolor("#e8f2ff")

    if title:
        ax.set_title(title, fontsize=10, pad=8)
    note = "* primary (soft-posterior) correction · Acc = diagonal P(correct) · "
    note += "matrices tagged ·g act in GMM-component space"
    ax.text(0.5, -0.02, note, transform=ax.transAxes, ha="center",
            va="top", fontsize=7.5, color="#555555")

    fig.tight_layout()
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Gaussian-fit sanity checks in RAW posterior-coordinate space (Hunt notes)
# ---------------------------------------------------------------------------

def _gmm_marginal_in_posterior(gmm, axis: int, alr_ref: int, n_classes: int,
                               n_samples: int = 200_000, random_state: int = 0):
    """Draw samples from the ALR-space GMM, map back to the simplex, and return
    the marginal value of posterior coordinate ``axis`` together with the
    component label of every sample. Used to overlay the *actual fitted model*
    (back-projected into raw posterior space) on the sanity histograms."""
    from .preprocess import inverse_alr

    rng = np.random.default_rng(random_state)
    K = gmm.n_components
    counts = rng.multinomial(n_samples, gmm.weights_)
    vals, comp = [], []
    for k in range(K):
        if counts[k] == 0:
            continue
        y = rng.multivariate_normal(gmm.means_[k], _get_full_cov(gmm, k), size=counts[k])
        p = inverse_alr(y, ref=alr_ref, n_classes=n_classes)
        vals.append(p[:, axis])
        comp.append(np.full(counts[k], k))
    return np.concatenate(vals), np.concatenate(comp)


def plot_axis_gaussian_sanity(
    posterior: np.ndarray,
    hard_labels: np.ndarray,
    labels,
    axis: int = 0,
    subsets=None,
    gmm=None,
    alr_ref: int = -1,
    bins: int = 60,
    out=None,
    random_state: int = 0,
):
    """1-D Gaussian sanity check along one posterior coordinate (e.g. the P6 axis).

    For every requested class *subset* this draws the histogram of the chosen
    posterior coordinate together with:

    * a per-class empirical Gaussian fit (mean/std of that coordinate within the
      class), weighted by the class fraction, and their **mixture sum** — so you
      can check whether the component peaks line up with the data peaks;
    * a single Gaussian fit to the whole subset (dashed grey);
    * (optional) the **back-projected fitted GMM** marginal (black dotted), i.e.
      the actual model used for the confusion matrix mapped from ALR space into
      raw posterior space.

    Parameters
    ----------
    posterior   : (N, K) protein-only posteriors (rows sum to 1).
    hard_labels : (N,) integer class index in ``[0, K)``.
    subsets     : list of ``(title, tuple_of_class_indices)``. Defaults to each
                  individual class, the all-but-middle pair (K==3), and all
                  classes combined — matching the Hunt notes (6, 8, 6+8, 6+7+8).
    """
    from scipy.stats import norm

    K = posterior.shape[1]
    axis_label = labels[axis]
    if subsets is None:
        subsets = [(labels[k], (k,)) for k in range(K)]
        subsets.append(("+".join(labels) + "  (all classes)", tuple(range(K))))

    gmm_vals = gmm_comp = None
    if gmm is not None:
        gmm_vals, gmm_comp = _gmm_marginal_in_posterior(
            gmm, axis=axis, alr_ref=alr_ref, n_classes=K, random_state=random_state)

    # Zoom x-axis to the actual data range (not the full 0–1 simplex).
    # Use the global 0.5th–99.5th percentile so all panels share the same window.
    _all_v = posterior[:, axis]
    _p1, _p99 = np.percentile(_all_v, [0.5, 99.5])
    _pad = max((_p99 - _p1) * 0.3, 0.02)
    xlo = max(0.0, _p1 - _pad)
    xhi = min(1.0, _p99 + _pad)

    n = len(subsets)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 4.0 * nrow),
                             squeeze=False)
    axes_flat = axes.ravel()
    grid = np.linspace(xlo, xhi, 400)

    for s_idx, (stitle, classes) in enumerate(subsets):
        ax = axes_flat[s_idx]
        mask = np.isin(hard_labels, classes)
        v = posterior[mask, axis]
        if len(v) == 0:
            ax.set_title(f"{stitle}\n(no particles)")
            continue
        ax.set_title(f"{stitle}  (N={len(v):,})", fontsize=10)
        ax.hist(v, bins=bins, range=(0, 1), density=True,
                color="#bbbbbb", alpha=0.7)

        # per-class component Gaussians + mixture sum
        mixture = np.zeros_like(grid)
        n_tot = len(v)
        for k in classes:
            vk = posterior[hard_labels == k, axis]
            if len(vk) < 2:
                continue
            frac = len(vk) / n_tot
            mu, sd = vk.mean(), vk.std(ddof=1)
            if sd <= 0:
                continue
            comp_pdf = frac * norm.pdf(grid, mu, sd)
            mixture += comp_pdf
            ax.plot(grid, comp_pdf, color=_COLORS[k % len(_COLORS)],
                    lw=1.8, label=f"{labels[k]}  μ={mu:.2f} σ={sd:.2f}")
            ax.axvline(mu, color=_COLORS[k % len(_COLORS)], ls=":", lw=0.8)
        if len(classes) > 1:
            ax.plot(grid, mixture, color="black", lw=2.0, label="component mixture")

        # single Gaussian fit to the whole subset — only shown for multi-class
        # panels (for a single class it is identical to the component curve)
        if len(classes) > 1:
            mu_all, sd_all = v.mean(), v.std(ddof=1)
            if sd_all > 0:
                ax.plot(grid, norm.pdf(grid, mu_all, sd_all), color="0.35",
                        ls="--", lw=1.4, label=f"single fit μ={mu_all:.2f}")

        # back-projected fitted GMM marginal (model that drives confusion)
        if gmm_vals is not None:
            gm = np.isin(gmm_comp, classes)
            if gm.any():
                ax.hist(gmm_vals[gm], bins=bins, range=(xlo, xhi), density=True,
                        histtype="step", color="black", ls=":", lw=1.6,
                        label="fitted GMM")

        ax.set_xlabel(f"{axis_label} class probability", fontsize=9)
        ax.set_ylabel("density", fontsize=9)
        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
        ax.set_xlim(xlo, xhi)
        # vertical reference line at 1/K (uniform probability)
        ax.axvline(1.0 / K, color="0.6", ls="-", lw=0.8, zorder=0)
        # annotate the uniform line on the last (all-classes) panel only
        if stitle.endswith("(all classes)"):
            ax.text(1.0 / K + 0.002, ax.get_ylim()[1] * 0.92,
                    f"uniform\n(1/{K}={1/K:.2f})", fontsize=7, color="0.45",
                    va="top")

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(
        f"{axis_label} class probability \u2014 how well are the classes separated?\n"
        f"If perfectly separated, {axis_label}-assigned particles would peak near 1.0.  "
        f"The grey vertical line marks 1/{K} = perfectly mixed (no separation).",
        fontsize=9, color="0.25",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig


def _draw_ellipse_outline(ax, mean, cov, n_std_list, color, ls="-", lw=1.6):
    """Draw unfilled confidence-ellipse outlines (for model-vs-data overlays)."""
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)
    angle = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
    for ns in n_std_list:
        w, h = 2.0 * ns * np.sqrt(vals)
        ell = Ellipse(xy=mean, width=w, height=h, angle=angle,
                      facecolor="none", edgecolor=color, ls=ls, lw=lw, zorder=6)
        ell.set_clip_box(ax.bbox)
        ax.add_patch(ell)


def plot_posterior_space_gmm(
    posterior: np.ndarray,
    hard_labels: np.ndarray,
    labels,
    gmm=None,
    alr_ref: int = -1,
    n_std=(1, 2),
    max_points: int = 8000,
    out=None,
    random_state: int = 0,
):
    """2-D Gaussian sanity check in RAW posterior-coordinate space.

    One panel per pair of protein axes. Each panel shows the particle scatter in
    raw posterior coordinates coloured by hard class, the **empirical** per-class
    Gaussian ellipse (solid), and — back-projected from ALR space — the
    **fitted GMM** per-component ellipse (dashed). Agreement between solid and
    dashed ellipses is the sanity check that the model fit matches the data.
    """
    from itertools import combinations

    rng = np.random.default_rng(random_state)
    K = posterior.shape[1]
    pairs = list(combinations(range(K), 2))

    # back-projected GMM samples per component -> empirical mean/cov in raw space
    gmm_stats = None
    if gmm is not None:
        from .preprocess import inverse_alr
        gmm_stats = {}
        for k in range(gmm.n_components):
            y = rng.multivariate_normal(gmm.means_[k], _get_full_cov(gmm, k), size=40_000)
            p = inverse_alr(y, ref=alr_ref, n_classes=K)
            gmm_stats[k] = (p.mean(axis=0), np.cov(p.T))

    n = len(pairs)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.8 * nrow),
                             squeeze=False)
    axes_flat = axes.ravel()

    for p_idx, (a, b) in enumerate(pairs):
        ax = axes_flat[p_idx]
        for k in range(K):
            m = hard_labels == k
            idx = np.where(m)[0]
            if len(idx) > max_points:
                idx = rng.choice(idx, max_points, replace=False)
            ax.scatter(posterior[idx, a], posterior[idx, b], s=3, alpha=0.10,
                       color=_COLORS[k % len(_COLORS)], rasterized=True)
            # empirical per-class ellipse (solid)
            pts = posterior[m][:, [a, b]]
            if len(pts) > 2:
                _draw_ellipse_outline(ax, pts.mean(axis=0), np.cov(pts.T),
                                      n_std, color=_COLORS[k % len(_COLORS)],
                                      ls="-", lw=2.2)
            # fitted-GMM back-projected ellipse (dashed)
            if gmm_stats is not None:
                gm, gc = gmm_stats[k]
                _draw_ellipse_outline(ax, gm[[a, b]], gc[np.ix_([a, b], [a, b])],
                                      n_std, color=_COLORS[k % len(_COLORS)],
                                      ls="--", lw=1.6)
        # zoom to the data region (clamped to the simplex) so the small
        # raw-space ellipses are legible instead of cramped in [0, 1]^2
        lo = np.maximum(0.0, posterior[:, [a, b]].min(axis=0) - 0.05)
        hi = np.minimum(1.0, posterior[:, [a, b]].max(axis=0) + 0.05)
        ax.set_xlabel(f"{labels[a]} posterior")
        ax.set_ylabel(f"{labels[b]} posterior")
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_title(f"{labels[a]} vs {labels[b]}", fontsize=10)

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    handles = [Patch(facecolor=_COLORS[k % len(_COLORS)], label=labels[k])
               for k in range(K)]
    handles += [
        Line2D([0], [0], color="0.2", ls="-", lw=2.0, label="empirical"),
        Line2D([0], [0], color="0.2", ls="--", lw=1.5, label="fitted GMM"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=8, frameon=False)
    fig.suptitle("2-D Gaussian sanity check in raw posterior space  "
                 "(solid = empirical · dashed = fitted GMM)", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig
