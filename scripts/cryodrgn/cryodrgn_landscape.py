#!/usr/bin/env python
"""Visualise the cryoDRGN conformational landscape with the latent GMM on top.

The latent space is multi-dimensional (zdim 8-10), so we project it down to a
2-D plane to *look* at it. PCA is used for the main view because it is linear:
a Gaussian stays a Gaussian under PCA, so we can draw each latent-GMM component
as an honest ellipse (the projection of its multi-dimensional bell curve into
the plane). UMAP is added only as a shape-intuition scatter (UMAP warps
distances, so ellipses would be meaningless there -- we don't draw them on it).

What you get (results dir):
  - latent_landscape.png : 4 panels --
      (A) PCA density of the whole landscape + GMM components as 1/2-sigma
          ellipses, labelled with their aligned CryoSPARC class (P6/P7/P8);
      (B) same plane, points coloured by CryoSPARC class;
      (C) same plane, points coloured by GMM component (hard assignment);
      (D) 1-D density along PC1 (the dominant axis) with each GMM component's
          projected bell curve + their sum -- the clearest "one cloud vs.
          separate bumps" picture.
  - latent_landscape_umap.png : UMAP scatter coloured by GMM component (shape
    only, no ellipses).
  - latent_landscape_fine.png : same as panel A but with K = BIC-ish many
    components, so you can see ALL the sub-groups the model could split into.

Run with the cryoDRGN env from repo root::

    python scripts/cryodrgn/cryodrgn_landscape.py \
      --z results_cryodrgn/J1442_real/train_z10/z.100.pkl \
      --passthrough-cs data/cryosparc_P25_J1442_passthrough_particles_all_classes.cs \
      --cs data/cryosparc_P25_J1442_00000_particles.cs \
      --n-dummies 6 --protein-idx 6 7 8 -k 3 --k-fine 8 \
      -o results_cryodrgn/J1442_real/landscape_z10
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for p in (_REPO, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Reuse the loading/alignment already written for the analysis script.
import cryodrgn_latent_gmm as clg


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def project_gaussian(mean, cov, pca):
    """Project a d-dim Gaussian (mean, cov) into the 2-D PCA plane.

    scores = (x - pca.mean_) @ V.T, with V = pca.components_ (2 x d).
    => mean2 = pca.transform(mean), cov2 = V @ cov @ V.T  (exact for linear map).
    """
    V = pca.components_
    mean2 = pca.transform(np.asarray(mean).reshape(1, -1))[0]
    cov2 = V @ cov @ V.T
    return mean2, cov2


def draw_ellipse(ax, mean2, cov2, color, label=None, nsigs=(1.0, 2.0)):
    vals, vecs = np.linalg.eigh(cov2)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    for i, ns in enumerate(sorted(nsigs)):
        w, h = 2 * ns * np.sqrt(np.maximum(vals, 1e-12))
        e = Ellipse(mean2, w, h, angle=angle, fill=False,
                    edgecolor=color, lw=2.2 if i == 0 else 1.4,
                    ls="-" if i == 0 else "--", alpha=0.95)
        ax.add_patch(e)
    ax.plot(*mean2, marker="*", ms=14, color=color, mec="black", mew=0.6,
            zorder=5)
    if label:
        ax.annotate(label, mean2, textcoords="offset points", xytext=(8, 8),
                    fontsize=11, fontweight="bold", color=color,
                    path_effects=None)


def landscape_panel(ax, scores, gmm_xs, pca, labels=None, palette=None):
    """Panel A / fine: hexbin density + projected GMM ellipses."""
    hb = ax.hexbin(scores[:, 0], scores[:, 1], gridsize=70, cmap="Greys",
                   bins="log", mincnt=1)
    K = len(gmm_xs.means_)
    palette = palette or plt.cm.tab10(np.linspace(0, 1, max(K, 3)))
    for i in range(K):
        m2, c2 = project_gaussian(gmm_xs.means_[i], gmm_xs.covariances_[i], pca)
        lab = labels[i] if labels else f"g{i}  ({gmm_xs.weights_[i]*100:.0f}%)"
        draw_ellipse(ax, m2, c2, palette[i % len(palette)], label=lab)
    ax.set_xlabel("PC1 (dominant conformational axis)")
    ax.set_ylabel("PC2")
    return hb


def pc1_marginal(ax, scores, gmm_xs, pca, palette=None, labels=None):
    """Panel D: 1-D density along PC1 + each component's projected bell curve."""
    x = scores[:, 0]
    lo, hi = np.percentile(x, [0.2, 99.8])
    grid = np.linspace(lo, hi, 400)
    ax.hist(x, bins=160, range=(lo, hi), density=True, color="0.8",
            edgecolor="none", label="all particles")
    K = len(gmm_xs.means_)
    palette = palette or plt.cm.tab10(np.linspace(0, 1, max(K, 3)))
    total = np.zeros_like(grid)
    w1 = pca.components_[0]  # PC1 direction in standardized space
    for i in range(K):
        mu1 = pca.transform(gmm_xs.means_[i].reshape(1, -1))[0, 0]
        var1 = float(w1 @ gmm_xs.covariances_[i] @ w1)
        wgt = gmm_xs.weights_[i]
        pdf = wgt * np.exp(-0.5 * (grid - mu1) ** 2 / var1) / np.sqrt(2 * np.pi * var1)
        total += pdf
        leg = labels[i] if labels else f"comp {i} ({wgt*100:.0f}%)"
        ax.plot(grid, pdf, color=palette[i % len(palette)], lw=2, label=leg)
    ax.plot(grid, total, color="black", lw=2.2, ls="--", label="GMM sum")
    ax.set_xlabel("PC1 score")
    ax.set_ylabel("density")
    ax.legend(fontsize=8, loc="upper right")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--z", required=True)
    ap.add_argument("--passthrough-cs", required=True)
    ap.add_argument("--cs", required=True)
    ap.add_argument("--n-dummies", type=int, default=6)
    ap.add_argument("--protein-idx", type=int, nargs="+", default=[6, 7, 8])
    ap.add_argument("-k", type=int, default=3,
                    help="components for the labelled landscape (matches CryoSPARC classes)")
    ap.add_argument("--k-fine", type=int, default=8,
                    help="components for the 'all sub-groups' fine view")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--umap", action="store_true",
                    help="also render a UMAP scatter (slower; needs umap-learn)")
    ap.add_argument("-o", "--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # 1) load + align (reuse the analysis script's logic)
    z = clg.load_latent(args.z)
    z_a, cryo_post, cryo_hard, uid_a, n_prot = clg.align_z_to_posteriors(
        z, args.passthrough_cs, args.cs, args.n_dummies, args.protein_idx)

    # 2) standardize, PCA(2) for the viewing plane
    scaler = StandardScaler().fit(z_a)
    Xs = scaler.transform(z_a)
    pca = PCA(n_components=2, random_state=args.seed).fit(Xs)
    scores = pca.transform(Xs)
    evr = pca.explained_variance_ratio_
    print(f"[pca] PC1/PC2 explain {evr[0]*100:.1f}% / {evr[1]*100:.1f}% "
          f"of standardized latent variance")

    # 3) GMM (K labelled) on standardized z, aligned to CryoSPARC classes
    gmm = GaussianMixture(args.k, covariance_type="full", reg_covar=1e-6,
                          max_iter=500, tol=1e-5, n_init=10,
                          random_state=args.seed).fit(Xs)
    resp = gmm.predict_proba(Xs)
    # align_components returns (perm, T) where drgn[:, perm] matches CryoSPARC,
    # i.e. CryoSPARC class j <-> GMM component perm[j]. We want the inverse:
    # for GMM component i, its CryoSPARC class is inv_perm[i].
    perm, _T = clg.align_components(cryo_post, resp)
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(len(perm))
    class_names = [f"P{j}" for j in args.protein_idx]
    labels = [f"{class_names[inv_perm[i]]} ({gmm.weights_[i]*100:.0f}%)"
              for i in range(args.k)]
    # consistent colour per CryoSPARC class
    class_colors = plt.cm.Set1(np.linspace(0, 1, max(len(class_names), 3)))
    comp_colors = [class_colors[inv_perm[i]] for i in range(args.k)]

    hard_gmm = resp.argmax(1)

    # compute component separation for annotation
    sep_stats = clg.separation_stats(gmm)
    min_sep = sep_stats["min_separation_sd"]
    mean_sep = sep_stats["mean_separation_sd"]
    print(f"[sep] min separation {min_sep:.2f} SD, mean {mean_sep:.2f} SD")

    # write GMM component means as z-vectors for cryodrgn eval_vol
    # inverse-transform from standardized space back to original z space
    peaks_z = scaler.inverse_transform(gmm.means_)  # (K, zdim)
    peaks_order = [class_names[inv_perm[i]] for i in range(args.k)]
    z_peaks_path = os.path.join(args.outdir, "z_gmm_peaks.txt")
    np.savetxt(z_peaks_path, peaks_z, fmt="%.10e")
    print(f"[peaks] wrote GMM peak z-vectors -> {z_peaks_path}")
    print(f"        row order (class alignment): {peaks_order}")
    print(f"        eval_vol command (run on cluster):")
    print(f"          cryodrgn eval_vol <workdir>/weights.pkl \\")
    print(f"            --config <workdir>/config.yaml \\")
    print(f"            --zfile {z_peaks_path} \\")
    print(f"            --Apix 0.83 \\")
    print(f"            -o {os.path.join(args.outdir, 'peak_volumes')}")

    # 4) main 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axA, axB, axC, axD = axes.ravel()

    hb = landscape_panel(axA, scores, gmm, pca, labels=labels, palette=comp_colors)
    axA.set_title("A. Conformational landscape (PCA density) + GMM components")
    cb = fig.colorbar(hb, ax=axA, fraction=0.046, pad=0.04)
    cb.set_label("log10(particle count)")

    for j, name in enumerate(class_names):
        m = cryo_hard == j
        axB.scatter(scores[m, 0], scores[m, 1], s=2, alpha=0.25,
                    color=class_colors[j], label=name, rasterized=True)
    axB.set_title("B. Coloured by CryoSPARC class")
    axB.set_xlabel("PC1"); axB.set_ylabel("PC2")
    lgB = axB.legend(markerscale=6, fontsize=10)
    for h in lgB.legend_handles:
        h.set_alpha(1)

    for i in range(args.k):
        m = hard_gmm == i
        axC.scatter(scores[m, 0], scores[m, 1], s=2, alpha=0.25,
                    color=comp_colors[i], label=labels[i], rasterized=True)
    axC.set_title("C. Coloured by latent-GMM component")
    axC.set_xlabel("PC1"); axC.set_ylabel("PC2")
    lgC = axC.legend(markerscale=6, fontsize=10)
    for h in lgC.legend_handles:
        h.set_alpha(1)

    pc1_marginal(axD, scores, gmm, pca, palette=comp_colors, labels=labels)
    axD.set_title(
        f"D. PC1 marginal density with GMM bell curves\n"
        f"(min component separation: {min_sep:.2f} SD, mean: {mean_sep:.2f} SD)")

    # annotate separation on panel A
    axA.set_title(
        f"A. Conformational landscape (PCA density) + GMM components\n"
        f"min sep: {min_sep:.2f} SD  |  mean sep: {mean_sep:.2f} SD  "
        f"(>2 = discrete, <2 = continuous)")

    fig.suptitle(
        f"cryoDRGN conformational landscape  |  {len(z_a):,} particles  "
        f"|  zdim {z_a.shape[1]}  |  PC1+PC2 = {(evr[0]+evr[1])*100:.0f}% of variance",
        fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_main = os.path.join(args.outdir, "latent_landscape.png")
    fig.savefig(out_main, dpi=150)
    plt.close(fig)
    print(f"[plot] {out_main}")

    # individual panel PNGs
    def _save_panel(panel_fn, out_name, title, extra_kw=None):
        fig1, ax1 = plt.subplots(figsize=(8, 7))
        panel_fn(ax1, **(extra_kw or {}))
        ax1.set_title(title, fontsize=12)
        fig1.tight_layout()
        fig1.savefig(os.path.join(args.outdir, out_name), dpi=150)
        plt.close(fig1)
        print(f"[plot] {out_name}")

    # Panel A
    figA, axA2 = plt.subplots(figsize=(8, 7))
    hb2 = landscape_panel(axA2, scores, gmm, pca, labels=labels, palette=comp_colors)
    axA2.set_title(
        f"Conformational landscape (PCA density) + GMM components\n"
        f"min sep: {min_sep:.2f} SD  |  mean sep: {mean_sep:.2f} SD  "
        f"(>2 = discrete, <2 = continuous)", fontsize=11)
    cb2 = figA.colorbar(hb2, ax=axA2, fraction=0.046, pad=0.04)
    cb2.set_label("log10(particle count)")
    figA.tight_layout()
    figA.savefig(os.path.join(args.outdir, "panel_A_landscape.png"), dpi=150)
    plt.close(figA)
    print("[plot] panel_A_landscape.png")

    # Panel B
    figB, axB2 = plt.subplots(figsize=(8, 7))
    for j, name in enumerate(class_names):
        m = cryo_hard == j
        axB2.scatter(scores[m, 0], scores[m, 1], s=2, alpha=0.25,
                     color=class_colors[j], label=name, rasterized=True)
    axB2.set_title("Coloured by CryoSPARC class", fontsize=12)
    axB2.set_xlabel("PC1"); axB2.set_ylabel("PC2")
    lgB2 = axB2.legend(markerscale=6, fontsize=10)
    for h in lgB2.legend_handles:
        h.set_alpha(1)
    figB.tight_layout()
    figB.savefig(os.path.join(args.outdir, "panel_B_cryosparc_class.png"), dpi=150)
    plt.close(figB)
    print("[plot] panel_B_cryosparc_class.png")

    # Panel C
    figC, axC2 = plt.subplots(figsize=(8, 7))
    for i in range(args.k):
        m = hard_gmm == i
        axC2.scatter(scores[m, 0], scores[m, 1], s=2, alpha=0.25,
                     color=comp_colors[i], label=labels[i], rasterized=True)
    axC2.set_title("Coloured by latent-GMM component", fontsize=12)
    axC2.set_xlabel("PC1"); axC2.set_ylabel("PC2")
    lgC2 = axC2.legend(markerscale=6, fontsize=10)
    for h in lgC2.legend_handles:
        h.set_alpha(1)
    figC.tight_layout()
    figC.savefig(os.path.join(args.outdir, "panel_C_gmm_component.png"), dpi=150)
    plt.close(figC)
    print("[plot] panel_C_gmm_component.png")

    # Panel D
    figD, axD2 = plt.subplots(figsize=(8, 5))
    pc1_marginal(axD2, scores, gmm, pca, palette=comp_colors, labels=labels)
    axD2.set_title(
        f"PC1 marginal density with GMM bell curves\n"
        f"min component separation: {min_sep:.2f} SD  |  mean: {mean_sep:.2f} SD",
        fontsize=11)
    figD.tight_layout()
    figD.savefig(os.path.join(args.outdir, "panel_D_pc1_marginal.png"), dpi=150)
    plt.close(figD)
    print("[plot] panel_D_pc1_marginal.png")

    # 5) fine view: more components = "all sub-groups it could split into"
    if args.k_fine and args.k_fine > args.k:
        gmm_f = GaussianMixture(args.k_fine, covariance_type="full",
                                reg_covar=1e-6, max_iter=500, tol=1e-5,
                                n_init=6, random_state=args.seed).fit(Xs)
        figf, axf = plt.subplots(figsize=(9, 8))
        landscape_panel(axf, scores, gmm_f, pca)
        axf.set_title(f"Landscape with K={args.k_fine} components "
                      f"(finer sub-grouping)")
        figf.tight_layout()
        out_fine = os.path.join(args.outdir, "latent_landscape_fine.png")
        figf.savefig(out_fine, dpi=150)
        plt.close(figf)
        print(f"[plot] {out_fine}")

    # 6) optional UMAP scatter (shape only)
    if args.umap:
        try:
            import umap
            emb = umap.UMAP(n_neighbors=30, min_dist=0.3,
                            random_state=args.seed).fit_transform(Xs)
            figu, (u1, u2) = plt.subplots(1, 2, figsize=(15, 6.5))
            for j, name in enumerate(class_names):
                m = cryo_hard == j
                u1.scatter(emb[m, 0], emb[m, 1], s=2, alpha=0.25,
                           color=class_colors[j], label=name, rasterized=True)
            u1.set_title("UMAP coloured by CryoSPARC class")
            u1.legend(markerscale=6)
            for i in range(args.k):
                m = hard_gmm == i
                u2.scatter(emb[m, 0], emb[m, 1], s=2, alpha=0.25,
                           color=comp_colors[i], label=labels[i], rasterized=True)
            u2.set_title("UMAP coloured by latent-GMM component")
            u2.legend(markerscale=6)
            figu.suptitle("UMAP (shape only -- distances warped, no ellipses)",
                          fontsize=13)
            figu.tight_layout()
            out_umap = os.path.join(args.outdir, "latent_landscape_umap.png")
            figu.savefig(out_umap, dpi=150)
            plt.close(figu)
            print(f"[plot] {out_umap}")
        except Exception as e:  # pragma: no cover
            print(f"[umap] skipped: {e}")

    print(f"[done] wrote landscape figures to {args.outdir}")


if __name__ == "__main__":
    main()
