#!/usr/bin/env python
"""Reproduce the cryoDRGN paper's latent-space analysis figures (non-map panels).

Follows Zhong, Bepler, Berger & Davis, Nature Methods 18, 176-185 (2021),
"CryoDRGN: reconstruction of heterogeneous cryo-EM structures using neural
networks", reproducing every figure panel that visualizes the LATENT SPACE
(i.e. everything except the 3D density-map renderings, which require a GPU
decoder and ChimeraX):

  * Fig. 3e-h  -- latent-encoding distribution (1-D histogram of PC1; for a true
                  1-D model this is the raw z histogram). Modes/clusters = states.
  * Fig. 4c,e  -- PCA projection of the latent space with explained-variance (EV)
                  axis labels, exactly as the paper annotates "PC1 (EV, 0.16)".
  * Fig. 5a    -- 1-D latent histogram with the junk-filter cutoff drawn.
  * Fig. 5b,f  -- UMAP embedding (umap defaults k=15 neighbours, min_dist=0.1,
                  the paper's exact settings), density-coloured and class-coloured.
  * Fig. 5c,d  -- latent (PCA + UMAP) coloured by the existing CryoSPARC class
                  assignment, with the on-data mean encoding per class marked
                  (z_hat_M = mean of z over particles in class M, paper's Eq.).
  * Fig. 5c filtering -- five-component full-covariance GMM fit to the latent
                  (scikit-learn), |z| magnitude shown, the outlier (junk) cluster
                  highlighted. This is the paper's impurity-removal step.
  * Fig. 2c,d  -- training loss curve over epochs, parsed from run.log (optional).

It also writes ``kmeans_centers.txt`` (on-data k-means cluster centres, paper's
representative-sampling step) so you can feed them straight to ``cryodrgn
eval_vol`` to render the representative maps.

The latent z.pkl is produced on the GPU cluster by ``cryodrgn train_vae`` (see
results_cryodrgn/J64_real/RUNBOOK.md). Until the J64 run exists, validate the
script on an existing run, e.g. the J1497 latent.

Run with the cryoDRGN env (numpy/sklearn/scipy/matplotlib/umap) from repo root::

    python scripts/cryodrgn/cryodrgn_paper_figures.py \
      --z results_cryodrgn/J64_real/train_filtered/z.49.pkl \
      --cs data/J64/cryosparc_P25_J64_00102_particles.cs \
      --passthrough-cs data/J64/cryosparc_P25_J64_passthrough_particles.cs \
      --protein-idx <protein class indices> --n-dummies <#dummies> \
      --run-log results_cryodrgn/J64_real/train_filtered/run.log \
      -o results_cryodrgn/J64_real/paper_figures
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# Reuse the codebase's tested latent loader + uid-based class alignment.
import cryodrgn_latent_gmm as clg

# Paper figure-5 / pilot defaults.
N_FILTER_COMPONENTS = 5      # "a five-component, full-covariance GMM"
KMEANS_K = 20                # pilot k-means (k=20) for representative sampling
UMAP_NEIGHBORS = 15          # umap default k=15
UMAP_MIN_DIST = 0.1          # paper's min_dist=0.1
SEED = 0


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
def pca_2d(z):
    """Return (coords Nx2, explained-variance-ratio [ev1, ev2]) like Fig. 4c,e."""
    p = PCA(n_components=min(z.shape[1], 10), random_state=SEED).fit(z)
    coords = p.transform(z)[:, :2]
    ev = p.explained_variance_ratio_
    return coords, (float(ev[0]), float(ev[1] if len(ev) > 1 else 0.0))


def umap_2d(z):
    """2-D UMAP with the paper's settings (k=15, min_dist=0.1). None if no umap."""
    try:
        import umap  # noqa: F401
    except Exception as exc:  # pragma: no cover
        print(f"[umap] not available ({exc}); skipping UMAP panels")
        return None
    import umap as umap_mod
    reducer = umap_mod.UMAP(n_neighbors=UMAP_NEIGHBORS, min_dist=UMAP_MIN_DIST,
                            n_components=2, random_state=SEED)
    print(f"[umap] embedding {z.shape[0]:,} points "
          f"(n_neighbors={UMAP_NEIGHBORS}, min_dist={UMAP_MIN_DIST}) ...")
    return reducer.fit_transform(z)


def _density_hexbin(ax, xy, xlabel, ylabel, title):
    hb = ax.hexbin(xy[:, 0], xy[:, 1], gridsize=60, bins="log", cmap="Greys",
                   mincnt=1, linewidths=0.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return hb


# --------------------------------------------------------------------------- #
# Figure panels
# --------------------------------------------------------------------------- #
def fig_pca(coords, ev, outdir):
    """Fig. 4c,e style: PCA latent projection coloured by particle density."""
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    hb = _density_hexbin(
        ax, coords, f"PC1 (EV, {ev[0]:.2f})", f"PC2 (EV, {ev[1]:.2f})",
        "Latent space (PCA)")
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("log$_{10}$ particle count")
    fig.tight_layout()
    path = os.path.join(outdir, "fig_pca_density.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[fig] {path}")


def fig_umap(emb, outdir):
    """Fig. 5b style: UMAP embedding coloured by particle density."""
    if emb is None:
        return
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    hb = _density_hexbin(ax, emb, "UMAP1", "UMAP2", "Latent space (UMAP)")
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("log$_{10}$ particle count")
    fig.tight_layout()
    path = os.path.join(outdir, "fig_umap_density.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[fig] {path}")


def fig_colored_by_class(coords, ev, emb, hard, class_names, outdir):
    """Fig. 5c,d style: latent coloured by CryoSPARC class + on-data class means."""
    n_panels = 2 if emb is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5.6 * n_panels, 5.0),
                             squeeze=False)
    cmap = plt.get_cmap("tab10")
    panels = [("PCA", coords, f"PC1 (EV, {ev[0]:.2f})", f"PC2 (EV, {ev[1]:.2f})")]
    if emb is not None:
        panels.append(("UMAP", emb, "UMAP1", "UMAP2"))

    for col, (name, xy, xl, yl) in enumerate(panels):
        ax = axes[0][col]
        for k, cname in enumerate(class_names):
            m = hard == k
            if not np.any(m):
                continue
            ax.scatter(xy[m, 0], xy[m, 1], s=2, alpha=0.25,
                       color=cmap(k % 10), label=cname, rasterized=True)
            # on-data mean encoding per class (paper's z_hat_M), marked as a star
            cx, cy = xy[m, 0].mean(), xy[m, 1].mean()
            ax.scatter([cx], [cy], s=160, marker="*", color=cmap(k % 10),
                       edgecolor="black", linewidth=0.8, zorder=5)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f"Latent ({name}) coloured by CryoSPARC class")
        if col == 0:
            lg = ax.legend(markerscale=4, fontsize=8, loc="upper right",
                           framealpha=0.9)
            for h in lg.legend_handles:
                h.set_alpha(1.0)
    fig.tight_layout()
    path = os.path.join(outdir, "fig_latent_by_class.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[fig] {path}")


def fig_pc1_histogram(coords, hard, class_names, outdir):
    """Fig. 3e-h / Fig. 5a style: 1-D latent (PC1) histogram; modes = states."""
    pc1 = coords[:, 0]
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.0), sharex=True)

    axes[0].hist(pc1, bins=120, color="0.4")
    axes[0].set_ylabel("particle count")
    axes[0].set_title("Latent PC1 distribution (modes = candidate states)")

    if hard is not None:
        cmap = plt.get_cmap("tab10")
        for k, cname in enumerate(class_names):
            m = hard == k
            if np.any(m):
                axes[1].hist(pc1[m], bins=120, histtype="step", linewidth=1.6,
                             color=cmap(k % 10), label=cname)
        axes[1].legend(fontsize=8)
    axes[1].set_xlabel("latent PC1")
    axes[1].set_ylabel("particle count")
    axes[1].set_title("PC1 distribution split by CryoSPARC class")
    fig.tight_layout()
    path = os.path.join(outdir, "fig_pc1_histogram.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[fig] {path}")


def fig_junk_filter(z, coords, ev, emb, outdir):
    """Fig. 5c filtering: 5-component full-cov GMM; flag the outlier (junk) cluster.

    The paper identifies the impurity cluster by the magnitude of the latent
    encoding (|z|); the GMM component whose members have the largest mean |z| is
    flagged as the candidate junk cluster to remove before high-res training.
    """
    gmm = GaussianMixture(N_FILTER_COMPONENTS, covariance_type="full",
                          reg_covar=1e-6, max_iter=500, n_init=3,
                          random_state=SEED).fit(z)
    labels = gmm.predict(z)
    zmag = np.linalg.norm(z, axis=1)
    # outlier cluster = component with the largest mean |z|
    comp_mean_mag = np.array([zmag[labels == c].mean() if np.any(labels == c)
                              else -np.inf for c in range(N_FILTER_COMPONENTS)])
    junk = int(np.argmax(comp_mean_mag))
    frac = float(np.mean(labels == junk))

    n_panels = 3 if emb is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.4 * n_panels, 5.0),
                             squeeze=False)
    cmap = plt.get_cmap("tab10")

    # panel 1: PCA coloured by |z| (the quantity used to ID the junk cluster)
    ax = axes[0][0]
    sc = ax.scatter(coords[:, 0], coords[:, 1], s=2, c=zmag, cmap="viridis",
                    alpha=0.4, rasterized=True)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04).set_label("|z|")
    ax.set_xlabel(f"PC1 (EV, {ev[0]:.2f})")
    ax.set_ylabel(f"PC2 (EV, {ev[1]:.2f})")
    ax.set_title("Latent magnitude |z|")

    # panel 2: PCA coloured by 5-component GMM, junk cluster outlined
    ax = axes[0][1]
    for c in range(N_FILTER_COMPONENTS):
        m = labels == c
        is_junk = c == junk
        ax.scatter(coords[m, 0], coords[m, 1], s=3 if not is_junk else 6,
                   alpha=0.3 if not is_junk else 0.8,
                   color=cmap(c % 10),
                   edgecolor="none" if not is_junk else "black",
                   label=f"comp {c}" + (" (JUNK)" if is_junk else ""),
                   rasterized=True)
    ax.set_xlabel(f"PC1 (EV, {ev[0]:.2f})")
    ax.set_ylabel(f"PC2 (EV, {ev[1]:.2f})")
    ax.set_title(f"5-component GMM filter\njunk = comp {junk} ({frac*100:.1f}%)")
    lg = ax.legend(markerscale=3, fontsize=8, loc="lower left")
    for h in lg.legend_handles:
        h.set_alpha(1.0)

    # panel 3: UMAP coloured by GMM (if available)
    if emb is not None:
        ax = axes[0][2]
        for c in range(N_FILTER_COMPONENTS):
            m = labels == c
            is_junk = c == junk
            ax.scatter(emb[m, 0], emb[m, 1], s=3 if not is_junk else 6,
                       alpha=0.3 if not is_junk else 0.8, color=cmap(c % 10),
                       edgecolor="none" if not is_junk else "black",
                       rasterized=True)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_title("5-component GMM filter (UMAP)")

    fig.tight_layout()
    path = os.path.join(outdir, "fig_junk_filter.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[fig] {path}  -> junk cluster = comp {junk} ({frac*100:.1f}% of particles)")
    return labels, junk


def kmeans_representatives(z, outdir, k=KMEANS_K):
    """Paper's representative-sampling step: k-means -> on-data cluster centres.

    Writes kmeans_centers.txt (the latent vector nearest each centroid) for
    direct use with ``cryodrgn eval_vol --zfile``.
    """
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(z)
    centers = km.cluster_centers_
    on_data = np.empty_like(centers)
    for c in range(k):
        members = z[km.labels_ == c]
        d = np.linalg.norm(members - centers[c], axis=1)
        on_data[c] = members[int(np.argmin(d))]
    path = os.path.join(outdir, "kmeans_centers.txt")
    np.savetxt(path, on_data, fmt="%.6f")
    print(f"[kmeans] wrote {k} on-data centres -> {path} "
          f"(feed to: cryodrgn eval_vol ... --zfile {os.path.basename(path)})")


def fig_training_curve(run_log, outdir):
    """Fig. 2c,d style: training loss vs epoch, parsed from run.log."""
    if not run_log or not os.path.exists(run_log):
        return
    epochs, losses = [], []
    pat = re.compile(r"Epoch\s*=?\s*(\d+).*?(?:total )?loss\s*=?\s*([0-9.]+)",
                     re.IGNORECASE)
    with open(run_log, "r", errors="ignore") as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                epochs.append(int(m.group(1)))
                losses.append(float(m.group(2)))
    if not epochs:
        print(f"[loss] no epoch/loss lines parsed from {run_log}")
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(epochs, losses, "o-", color="steelblue")
    ax.set_xlabel("epoch")
    ax.set_ylabel("total loss")
    ax.set_title("cryoDRGN training curve")
    fig.tight_layout()
    path = os.path.join(outdir, "fig_training_curve.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[fig] {path}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--z", required=True, help="cryoDRGN latent z.N.pkl")
    ap.add_argument("--cs", default=None,
                    help="CryoSPARC *_particles.cs for optional class colouring")
    ap.add_argument("--passthrough-cs", default=None,
                    help="passthrough .cs (uid order matching z); required with --cs")
    ap.add_argument("--protein-idx", type=int, nargs="*", default=None,
                    help="protein class indices for class colouring")
    ap.add_argument("--n-dummies", type=int, default=0,
                    help="number of leading dummy classes to drop")
    ap.add_argument("--run-log", default=None,
                    help="train_vae run.log for the training-curve panel")
    ap.add_argument("--no-umap", action="store_true", help="skip UMAP panels")
    ap.add_argument("-o", "--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    z = clg.load_latent(args.z)

    hard, class_names = None, None
    if args.cs and args.passthrough_cs:
        z, _cryo_post, hard, _uid, n_prot = clg.align_z_to_posteriors(
            z, args.passthrough_cs, args.cs, args.n_dummies, args.protein_idx)
        idx = args.protein_idx if args.protein_idx else list(range(n_prot))
        class_names = [f"P{i}" for i in idx]
        print(f"[class] coloured by CryoSPARC classes: {class_names}")

    coords, ev = pca_2d(z)
    emb = None if (args.no_umap or z.shape[1] < 2) else umap_2d(z)

    fig_pca(coords, ev, args.outdir)
    fig_umap(emb, args.outdir)
    fig_pc1_histogram(coords, hard, class_names, args.outdir)
    if hard is not None:
        fig_colored_by_class(coords, ev, emb, hard, class_names, args.outdir)
    fig_junk_filter(z, coords, ev, emb, args.outdir)
    kmeans_representatives(z, args.outdir)
    fig_training_curve(args.run_log, args.outdir)
    print(f"\n[done] figures -> {args.outdir}")


if __name__ == "__main__":
    main()
