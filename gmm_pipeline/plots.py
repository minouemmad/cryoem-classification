"""Lightweight diagnostic plots for the GMM uncertainty pipeline."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse, Patch
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
    ax.bar(x + 0.18, corrected, width=0.35, label="Corrected (analytical)", color="#3a7")
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
    ax.set_title("Conformational populations  (corrected = (Cᵀ)⁻¹ pi_obs,  analytical)")
    ax.legend()
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
# Compact class table: populations + MC accuracy (email-friendly)
# ---------------------------------------------------------------------------

def plot_class_table(
    labels,
    pi_obs, pi_corr, pi_corr_std,
    pi_corr_lo, pi_corr_hi,
    confusion_mc: np.ndarray,
    confusion_analytical: np.ndarray | None = None,
    title: str = "",
    out=None,
):
    """Single compact table: Class | Obs. | Corr. | ±Std | 95% CI | Multi-class Acc."""
    K = len(labels)
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * (K + 1) + 0.9))
    ax.axis("off")

    acc_matrix = confusion_analytical if confusion_analytical is not None else confusion_mc
    col_headers = ["Class", "Obs.", "Corr.", "±Std", "95% CI", "Multi-class Acc."]
    rows = []
    for i, lbl in enumerate(labels):
        rows.append([
            lbl,
            f"{pi_obs[i]:.1%}",
            f"{pi_corr[i]:.1%}",
            f"±{pi_corr_std[i]:.1%}",
            f"{pi_corr_lo[i]:.1%}–{pi_corr_hi[i]:.1%}",
            f"{acc_matrix[i, i]:.1%}",
        ])

    tbl = ax.table(
        cellText=rows,
        colLabels=col_headers,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.auto_set_column_width(list(range(len(col_headers))))
    tbl.scale(1, 1.9)

    header_bg = "#dddddd"
    for j in range(len(col_headers)):
        cell = tbl[(0, j)]
        cell.set_facecolor(header_bg)
        cell.set_text_props(color="black", fontweight="bold")
        cell.set_edgecolor("#aaaaaa")

    acc_cmap = plt.get_cmap("RdYlGn")
    for i in range(K):
        bg = "white" if i % 2 == 0 else "#f5f5f5"
        for j in range(len(col_headers)):
            cell = tbl[(i + 1, j)]
            cell.set_facecolor(bg)
            cell.set_edgecolor("#cccccc")
        acc_val = acc_matrix[i, i]
        cell = tbl[(i + 1, 5)]
        cell.set_facecolor(acc_cmap(acc_val))
        cell.set_text_props(
            color="white" if acc_val < 0.45 or acc_val > 0.85 else "black"
        )

    if title:
        ax.set_title(title, fontsize=10, pad=8)

    fig.tight_layout()
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Summary table: populations + confusion diagonal + Bhattacharyya
# ---------------------------------------------------------------------------

def plot_summary_table(
    labels,
    pi_obs, pi_corr, pi_corr_mean, pi_corr_std, pi_corr_lo, pi_corr_hi,
    confusion_mc: np.ndarray,
    confusion_analytical: np.ndarray,
    bhatt_overlap: np.ndarray,
    diag: dict,
    title: str = "",
    out=None,
):
    """Two-panel summary figure (black-and-white table styling).

    Top panel  — populations + MC and ERF classification accuracy.
    Bottom panel — Bhattacharyya overlap heatmap.
    """
    K = len(labels)
    fig = plt.figure(figsize=(max(11, K * 2.0), 6.5 + K * 0.25))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.0], hspace=0.55)

    # ---- top: population + accuracy table ----
    ax_top = fig.add_subplot(gs[0])
    ax_top.axis("off")

    col_headers = ["Class", "Obs.", "Corr.", "±Std", "95% CI", "MC acc.", "ERF acc.", "Max leak"]
    rows = []
    for i, lbl in enumerate(labels):
        mc_acc = confusion_mc[i, i]
        erf_acc = confusion_analytical[i, i]
        off = confusion_mc[i].copy()
        off[i] = -1
        worst_j = int(off.argmax())
        worst_val = confusion_mc[i, worst_j]
        ci_str = f"{pi_corr_lo[i]:.0%}–{pi_corr_hi[i]:.0%}"
        rows.append([
            lbl,
            f"{pi_obs[i]:.1%}",
            f"{pi_corr_mean[i]:.1%}",
            f"±{pi_corr_std[i]:.1%}",
            ci_str,
            f"{mc_acc:.1%}",
            f"{erf_acc:.1%}",
            f"{worst_val:.0%}→{labels[worst_j]}",
        ])

    tbl = ax_top.table(
        cellText=rows,
        colLabels=col_headers,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(list(range(len(col_headers))))
    tbl.scale(1, 1.6)

    # B&W header styling
    header_bg = "#dddddd"
    for j in range(len(col_headers)):
        cell = tbl[(0, j)]
        cell.set_facecolor(header_bg)
        cell.set_text_props(color="black", fontweight="bold")
        cell.set_edgecolor("#aaaaaa")

    acc_cmap = plt.get_cmap("RdYlGn")
    for i in range(len(rows)):
        row_idx = i + 1
        mc_acc = confusion_mc[i, i]
        erf_acc = confusion_analytical[i, i]
        bg = "white" if i % 2 == 0 else "#f5f5f5"
        for j in range(len(col_headers)):
            cell = tbl[(row_idx, j)]
            cell.set_facecolor(bg)
            cell.set_edgecolor("#cccccc")
        for col_idx, acc_val in [(5, mc_acc), (6, erf_acc)]:
            tbl[(row_idx, col_idx)].set_facecolor(acc_cmap(acc_val))
            tbl[(row_idx, col_idx)].set_text_props(
                color="white" if acc_val < 0.45 or acc_val > 0.85 else "black"
            )

    top_title = title or f"BIC={diag['bic']:.0f}  ·  {diag['n_iter']} iters"
    ax_top.set_title(top_title, fontsize=10, pad=6)

    # ---- bottom: Bhattacharyya overlap heatmap ----
    ax_bot = fig.add_subplot(gs[1])
    ax_bot.axis("off")

    overlap_cmap = plt.get_cmap("YlOrRd")
    bhatt_rows = []
    for i in range(K):
        bhatt_rows.append([f"{bhatt_overlap[i, j]:.2f}" for j in range(K)])

    tbl2 = ax_bot.table(
        cellText=bhatt_rows,
        rowLabels=labels,
        colLabels=labels,
        loc="center",
        cellLoc="center",
    )
    tbl2.auto_set_font_size(False)
    tbl2.set_fontsize(9.5)
    tbl2.scale(1, 1.6)

    for j in range(K):
        cell = tbl2[(0, j)]
        cell.set_facecolor(header_bg)
        cell.set_text_props(color="black", fontweight="bold")
        cell.set_edgecolor("#aaaaaa")

    for i in range(K):
        for j in range(K):
            val = float(bhatt_overlap[i, j])
            tbl2[(i + 1, j)].set_facecolor(overlap_cmap(val))
            tbl2[(i + 1, j)].set_text_props(color="white" if val > 0.55 else "black")
            tbl2[(i + 1, j)].set_edgecolor("#cccccc")

    ax_bot.set_title(
        "Bhattacharyya overlap  (0 = distinct · 1 = identical)",
        fontsize=10, pad=6,
    )

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig
