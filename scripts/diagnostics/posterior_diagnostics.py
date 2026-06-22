"""Posterior quality diagnostic figures for J1442 (K=3) and J1497 (K=5).

Generates all figures needed to present the posterior flatness problem to a PI.
Run:
    python posterior_diagnostics.py
Outputs written to:  diagnostics/
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import entropy

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "diagnostics"
OUT.mkdir(exist_ok=True)

N_DUMMY = 6

# ---------------------------------------------------------------------------
# Load raw posteriors from .cs files
# ---------------------------------------------------------------------------

def load_cs_posteriors(path: str, n_dummy: int = 6):
    cs = np.load(path, allow_pickle=True)
    post_full = np.asarray(cs["alignments3D_multi/class_posterior"], dtype=np.float64)
    post_full /= post_full.sum(axis=1, keepdims=True).clip(min=1e-12)
    K_full = post_full.shape[1]
    post_protein = post_full[:, n_dummy:]
    post_protein /= post_protein.sum(axis=1, keepdims=True).clip(min=1e-12)
    labels = [f"P{n_dummy + i}" for i in range(K_full - n_dummy)]
    return post_full, post_protein, labels


post_full_1442, post_1442, labels_1442 = load_cs_posteriors(
    DATA / "cryosparc_P25_J1442_00000_particles.cs"
)
post_full_1497, post_1497, labels_1497 = load_cs_posteriors(
    DATA / "gP25W6J1497_00000_particles.cs"
)

DATASETS = {
    "J1442  (K=3)": (post_1442, labels_1442, post_full_1442),
    "J1497  (K=5)": (post_1497, labels_1497, post_full_1497),
}

# ============================================================================
# Figure 1 — Max-posterior histogram
# Shows: nearly all particles have their winning class below random-chance level
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
fig.suptitle(
    "Figure 1 — Max posterior confidence per particle\n",
    fontsize=12, y=1.01,
)

for ax, (name, (post, labels, _)) in zip(axes, DATASETS.items()):
    K = post.shape[1]
    max_post = post.max(axis=1)
    random_floor = 1.0 / K
    conf_threshold = 0.5

    ax.hist(max_post, bins=80, color="#4c72b0", edgecolor="none", alpha=0.85,
            density=True)
    ax.axvline(random_floor, color="red", linestyle="--", linewidth=1.8,
               label=f"1/K = {random_floor:.3f}  (random floor)")
    ax.axvline(conf_threshold, color="green", linestyle="--", linewidth=1.8,
               label=f"0.5 threshold  (confident)")
    ax.axvline(max_post.mean(), color="orange", linestyle="-", linewidth=1.6,
               label=f"Mean = {max_post.mean():.3f}")

    pct_confident = (max_post > conf_threshold).mean() * 100
    ax.set_title(
        f"{name}\n",
        fontsize=10,
    )
    ax.set_xlabel("Max posterior", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend(fontsize=9)
    x_max = min(1.0, max_post.max() * 1.5)
    ax.set_xlim(0, x_max)

fig.tight_layout()
fig.savefig(OUT / "fig1_max_posterior_histogram.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig1_max_posterior_histogram.png")


# ============================================================================
# Figure 2 — Per-class mean posterior + spread (violin)
# Shows: all classes receive nearly identical average posterior probability
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "Figure 2 — Per-class posterior distribution across all particles\n"
    "(flat violins = classes indistinguishable)",
    fontsize=12, y=1.01,
)

for ax, (name, (post, labels, _)) in zip(axes, DATASETS.items()):
    K = post.shape[1]
    parts = ax.violinplot(
        [post[:, k] for k in range(K)],
        positions=range(K),
        showmedians=True,
        showextrema=False,
        widths=0.65,
    )
    for pc in parts["bodies"]:
        pc.set_facecolor("#4c72b0")
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(2)

    ax.axhline(1.0 / K, color="red", linestyle="--", linewidth=1.5,
               label=f"1/K = {1/K:.3f}")
    ax.set_xticks(range(K), labels, fontsize=10)
    ax.set_ylabel("Posterior probability", fontsize=10)
    ax.set_title(name, fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, ax.get_ylim()[1])

fig.tight_layout()
fig.savefig(OUT / "fig2_per_class_posterior_violin.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig2_per_class_posterior_violin.png")


# ============================================================================
# Figure 3 — Shannon entropy per particle
# Shows: actual entropy vs theoretical max (uniform) and theoretical min (certain)
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle(
    "Figure 3 — Classification entropy per particle\n"
    "(H_max = log(K) = fully uncertain,  H=0 = perfectly certain)",
    fontsize=12, y=1.01,
)

for ax, (name, (post, labels, _)) in zip(axes, DATASETS.items()):
    K = post.shape[1]
    H = entropy(post.T)          # Shannon entropy, nats
    H_max = np.log(K)            # uniform distribution entropy

    ax.hist(H, bins=80, color="#55a868", edgecolor="none", alpha=0.85, density=True)
    ax.axvline(H_max, color="red", linestyle="--", linewidth=1.8,
               label=f"H_max = {H_max:.2f}  (uniform, K={K})")
    ax.axvline(H.mean(), color="orange", linestyle="-", linewidth=1.6,
               label=f"Mean H = {H.mean():.2f}")

    pct_near_max = (H > 0.95 * H_max).mean() * 100
    ax.set_title(
        f"{name}\n"
        f"{pct_near_max:.1f}% of particles within 5% of max entropy",
        fontsize=10,
    )
    ax.set_xlabel("Shannon entropy H (nats)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend(fontsize=9)

fig.tight_layout()
fig.savefig(OUT / "fig3_classification_entropy.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig3_classification_entropy.png")


# ============================================================================
# Figures 4 / 5 — Pairwise posterior scatter with GMM overlay
#
# For each pair of protein classes (i, j) we plot every particle as a dot in
# (P_i, P_j) space, colored by CryoSPARC hard assignment.  On top we overlay,
# for each hard-assigned class, the EMPIRICAL Gaussian model fit directly in
# raw probability space:
#
#   * a large black-edged dot at the per-class mean (mu_i, mu_j)
#   * horizontal/vertical bars showing +/- 1 sigma along each axis
#   * a 1-sigma covariance ellipse  (~39% probability mass for a 2D Gaussian)
#   * a 2-sigma covariance ellipse  (~86% probability mass)
#
# This overlay is fit in the SAME space as the scatter (raw probabilities),
# so it directly answers "does the Gaussian model capture the structure I see?"
# This is the figure-format Dr. Hunt uses in his summary slide.
#
# Indexing note: CryoSPARC stores classes 0-indexed but its UI / Hunt's slides
# label them 1-indexed.  We show both:  "P6 / Class 7" etc.
# ============================================================================
from matplotlib.patches import Ellipse


def _ellipse_from_cov(mean, cov, n_std=1.0, **kwargs):
    """Return a matplotlib Ellipse patch representing the n-sigma contour
    of a 2D Gaussian with the given mean and covariance.
    """
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2.0 * n_std * np.sqrt(np.maximum(vals, 0))
    return Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)


def _fit_gauss_2d(points):
    """Empirical mean (2,) and covariance (2,2) of a 2D point cloud."""
    mu = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return mu, np.atleast_2d(cov)


def _draw_pairwise_panel(ax, post, hard, labels, i, j, cmap,
                         subsample, point_size=1.5, point_alpha=0.12,
                         show_legend=False, fontsize=10):
    K = post.shape[1]
    # Scatter all particles colored by hard assignment
    for k in range(K):
        mask = hard[subsample] == k
        ax.scatter(
            post[subsample][mask, i],
            post[subsample][mask, j],
            s=point_size, alpha=point_alpha,
            color=cmap(k),
            rasterized=True,
        )

    # Overlay empirical Gaussian per class (mean, std bars, 1- and 2-sigma ellipses)
    import matplotlib.colors as mcolors
    for k in range(K):
        mask = hard == k
        if mask.sum() < 5:
            continue
        pts = post[mask][:, [i, j]]
        mu, cov = _fit_gauss_2d(pts)
        color = cmap(k)
        r, g, b, _ = mcolors.to_rgba(color)
        dark = (r * 0.6, g * 0.6, b * 0.6, 1.0)

        # 2σ: dashed contour in darkened colour — drawn above scatter (high zorder)
        ell2 = _ellipse_from_cov(mu, cov, n_std=2.0,
                                 edgecolor=dark, facecolor="none",
                                 linewidth=1.6, linestyle=(0, (5, 3)),
                                 alpha=0.9, zorder=6)
        ax.add_patch(ell2)

        # 1σ: semi-transparent fill + bold edge in darkened colour
        ell1_fill = _ellipse_from_cov(mu, cov, n_std=1.0,
                                      edgecolor="none", facecolor=color,
                                      alpha=0.28, zorder=7)
        ax.add_patch(ell1_fill)
        ell1 = _ellipse_from_cov(mu, cov, n_std=1.0,
                                 edgecolor=dark, facecolor="none",
                                 linewidth=2.0, alpha=1.0, zorder=8)
        ax.add_patch(ell1)

        # ±1σ axis-aligned bars + white mean dot
        sx, sy = np.sqrt(np.diag(cov))
        ax.errorbar(mu[0], mu[1], xerr=sx, yerr=sy,
                    fmt="none", ecolor=dark,
                    elinewidth=1.1, capsize=2.5, zorder=9)
        ax.plot(mu[0], mu[1], "o", color="white",
                markeredgecolor=dark, markeredgewidth=1.5,
                markersize=6, zorder=10)
        # +/-1 sigma axis-aligned error bars (sqrt of diag of cov)
        sx, sy = np.sqrt(np.diag(cov))
        ax.errorbar(mu[0], mu[1], xerr=sx, yerr=sy,
                    fmt="o", color=color, ecolor="black",
                    markeredgecolor="black", markersize=7,
                    elinewidth=1.4, capsize=3, zorder=10)

    ax.set_xlabel(f"P({labels[i]})", fontsize=fontsize)
    ax.set_ylabel(f"P({labels[j]})", fontsize=fontsize)
    ax.set_title(f"{labels[i]}  vs  {labels[j]}", fontsize=fontsize)


def _dual_labels(labels, dummy_offset=N_DUMMY):
    """Build labels like 'P6 / Class 7' for plot legends / titles."""
    out = []
    for lab in labels:
        # lab is e.g. 'P6' -> 0-indexed class 6 -> 1-indexed Class 7
        idx0 = int(lab.lstrip("P"))
        out.append(f"{lab} / Class {idx0 + 1}")
    return out


# Colour scheme matching Hunt's figure exactly:
#   P6 / Class 7 (NBD1lessNarrow) = green
#   P7 / Class 8 (NBD1lessWide)   = red
#   P8 / Class 9 (Vshaped)        = blue
_J1442_COLORS = ["#2ca02c", "#d62728", "#1f77b4"]   # green, red, blue
def cmap_1442(k): return _J1442_COLORS[k % len(_J1442_COLORS)]

cmap = plt.get_cmap("tab10")   # kept for J1497 (K=5)

# ----- Figure 4a (J1442, K=3): 3-panel pairwise scatter with GMM overlay -----

post = post_1442
labels = labels_1442
hard = post.argmax(axis=1)
K = post.shape[1]
pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
subsample = np.random.default_rng(42).choice(
    len(post), size=min(30_000, len(post)), replace=False
)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "Figure 4 — Pairwise posterior scatter with empirical Gaussian overlay  [J1442, K=3]\n"
    "Dots = particles, colored by CryoSPARC hard assignment.   Ellipses = 1\u03c3 (solid) and 2\u03c3 (dashed) per class.\n"
    "White dot = per-class mean,  error bars = \u00b11\u03c3 in each axis.",
    fontsize=11, y=1.04,
)
for ax, (i, j) in zip(axes, pairs):
    _draw_pairwise_panel(ax, post, hard, labels, i, j, cmap_1442, subsample,
                         point_size=2.0, point_alpha=0.15)

dual = _dual_labels(labels)
handles = [plt.Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=cmap_1442(k), markeredgecolor="black",
                      markersize=8)
           for k in range(K)]
fig.legend(
    handles,
    labels,
    fontsize=9,
    loc="lower center",
    ncol=K,
    bbox_to_anchor=(0.5, -0.05)
)
fig.tight_layout()
fig.savefig(OUT / "fig4_pairwise_posterior_scatter_J1442.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig4_pairwise_posterior_scatter_J1442.png")


# ----- Figure 4b (J1442): each pair as a standalone larger panel ------------
for (i, j) in pairs:
    fig, ax = plt.subplots(figsize=(7, 6))
    _draw_pairwise_panel(ax, post, hard, labels, i, j, cmap_1442, subsample,
                         point_size=2.5, point_alpha=0.18, fontsize=11)
    dual = _dual_labels(labels)
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=cmap_1442(k), markeredgecolor="black",
                          markersize=8)
               for k in range(K)]
    ax.legend(handles, dual, title="Hard assignment", fontsize=9, loc="best")
    ax.set_title(
        f"J1442  —  {labels[i]} (Class {int(labels[i][1:]) + 1})  vs  "
        f"{labels[j]} (Class {int(labels[j][1:]) + 1})\n"
        "Per-class empirical Gaussian overlaid (mean dot, \u00b11\u03c3 bars, 1\u03c3/2\u03c3 ellipses)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(
        OUT / f"fig4_pair_J1442_{labels[i]}_vs_{labels[j]}.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    print(f"Saved fig4_pair_J1442_{labels[i]}_vs_{labels[j]}.png")


# ----- Figure 5a (J1497, K=5): full 10-pair grid with GMM overlay -----------

post = post_1497
labels = labels_1497
hard = post.argmax(axis=1)
K = post.shape[1]
pairs5 = [(i, j) for i in range(K) for j in range(i + 1, K)]
subsample = np.random.default_rng(42).choice(
    len(post), size=min(30_000, len(post)), replace=False
)

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle(
    "Figure 5 — Pairwise posterior scatter with empirical Gaussian overlay  [J1497, K=5]\n"
    "Ellipses = 1\u03c3 / 2\u03c3 contours per class.   Black-edged marker = mean, error bars = \u00b11\u03c3.",
    fontsize=11, y=1.02,
)
for ax, (i, j) in zip(axes.flat, pairs5):
    _draw_pairwise_panel(ax, post, hard, labels, i, j, cmap, subsample,
                         point_size=1.4, point_alpha=0.10, fontsize=8)
    ax.tick_params(labelsize=7)
for ax in axes.flat[len(pairs5):]:
    ax.set_visible(False)

dual = _dual_labels(labels)
handles = [plt.Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=cmap(k), markeredgecolor="black",
                      markersize=8)
           for k in range(K)]
fig.legend(handles, dual, title="Hard assignment  (0-idx / CryoSPARC UI idx)",
           fontsize=9, loc="lower right", bbox_to_anchor=(1.0, 0.0), ncol=1)
fig.tight_layout()
fig.savefig(OUT / "fig5_pairwise_posterior_scatter_J1497.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig5_pairwise_posterior_scatter_J1497.png")


# ----- Figure 5b (J1497): individual standalone panels for each pair --------
for (i, j) in pairs5:
    fig, ax = plt.subplots(figsize=(7, 6))
    _draw_pairwise_panel(ax, post, hard, labels, i, j, cmap, subsample,
                         point_size=2.0, point_alpha=0.15, fontsize=11)
    dual = _dual_labels(labels)
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=cmap(k), markeredgecolor="black",
                          markersize=8)
               for k in range(K)]
    ax.legend(handles, dual, title="Hard assignment", fontsize=8, loc="best")
    ax.set_title(
        f"J1497  —  {labels[i]} (Class {int(labels[i][1:]) + 1})  vs  "
        f"{labels[j]} (Class {int(labels[j][1:]) + 1})\n"
        "Per-class empirical Gaussian overlaid",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(
        OUT / f"fig5_pair_J1497_{labels[i]}_vs_{labels[j]}.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
print("Saved fig5_pair_J1497_*.png individual pair files")


# ============================================================================
# Figure 6 — Summary comparison table
# Shows key stats side by side for both datasets
# ============================================================================

fig, ax = plt.subplots(figsize=(11, 4))
ax.axis("off")
fig.suptitle(
    "Figure 6 — Posterior quality summary table",
    fontsize=13, y=0.97,
)

rows = []
for name, (post, labels, post_full) in DATASETS.items():
    K = post.shape[1]
    K_full = post_full.shape[1]
    max_post = post.max(axis=1)
    # margin = top-1 minus top-2 posterior
    sorted_post = np.sort(post, axis=1)
    margin = sorted_post[:, -1] - sorted_post[:, -2]
    H = entropy(post.T)
    H_max = np.log(K)
    random_floor = 1.0 / K
    pct_confident = (max_post > 0.5).mean() * 100
    pct_near_max_H = (H > 0.95 * H_max).mean() * 100
    rows.append([
        name,
        f"{K_full} ({K} protein + {K_full - K} dummy)",
        f"{len(post):,}",
        f"{random_floor:.3f}",
        f"{max_post.mean():.3f} ± {max_post.std():.3f}",
        f"{max_post.max():.3f}",
        f"{pct_confident:.1f}%",
        f"{margin.mean():.3f} ± {margin.std():.3f}",
        f"{H.mean():.3f} / {H_max:.3f}",
        f"{pct_near_max_H:.1f}%",
    ])

col_headers = [
    "Dataset", "Classes", "Particles",
    "1/K (random)", "Mean max posterior", "Best max posterior",
    "% confident (>0.5)",
    "Mean margin (1st−2nd)",
    "Mean H / H_max",
    "% near H_max",
]

tbl = ax.table(
    cellText=rows,
    colLabels=col_headers,
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.auto_set_column_width(list(range(len(col_headers))))
tbl.scale(1, 2.2)

header_bg = "#dddddd"
for j in range(len(col_headers)):
    cell = tbl[(0, j)]
    cell.set_facecolor(header_bg)
    cell.set_text_props(fontweight="bold")
    cell.set_edgecolor("#aaaaaa")

alert_cmap = plt.get_cmap("RdYlGn_r")
for i in range(len(rows)):
    bg = "white" if i % 2 == 0 else "#f5f5f5"
    for j in range(len(col_headers)):
        tbl[(i + 1, j)].set_facecolor(bg)
        tbl[(i + 1, j)].set_edgecolor("#cccccc")
    # colour the "% confident" cell: lower % = redder (more alarming)
    pct_val = float(rows[i][6].rstrip("%")) / 100
    tbl[(i + 1, 6)].set_facecolor(alert_cmap(1.0 - pct_val))
    tbl[(i + 1, 6)].set_text_props(color="white" if pct_val < 0.3 else "black")

fig.tight_layout()
fig.savefig(OUT / "fig6_posterior_quality_table.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig6_posterior_quality_table.png")


# ============================================================================
# Figure 7 — Stacked bar: mean posterior allocation per class
# Shows: each class gets ~equal share of probability mass → no separation
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "Figure 7 — Mean posterior allocated to each class, broken down by hard assignment\n"
    "(diagonal should dominate for well-separated classes)",
    fontsize=12, y=1.01,
)

cmap = plt.get_cmap("tab10")
for ax, (name, (post, labels, _)) in zip(axes, DATASETS.items()):
    K = post.shape[1]
    hard = post.argmax(axis=1)
    mean_post_matrix = np.zeros((K, K))
    for i in range(K):
        mask = hard == i
        if mask.sum() > 0:
            mean_post_matrix[i] = post[mask].mean(axis=0)

    x = np.arange(K)
    bottom = np.zeros(K)
    for j in range(K):
        vals = mean_post_matrix[:, j]
        ax.bar(x, vals, bottom=bottom,
               label=f"→ {labels[j]}", color=cmap(j), alpha=0.85, width=0.6)
        bottom += vals

    ax.axhline(1.0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xticks(x, [f"{labels[i]}\n(n={( hard==i).sum():,})" for i in range(K)], fontsize=9)
    ax.set_ylabel("Mean posterior probability", fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_title(name, fontsize=11)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    # diagonal annotation
    for i in range(K):
        diag_val = mean_post_matrix[i, i]
        ax.text(i, diag_val / 2 + mean_post_matrix[i, :i].sum(),
                f"{diag_val:.2f}", ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")

fig.tight_layout()
fig.savefig(OUT / "fig7_mean_posterior_stacked_bar.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig7_mean_posterior_stacked_bar.png")

print(f"\nAll figures written to {OUT.resolve()}/")
