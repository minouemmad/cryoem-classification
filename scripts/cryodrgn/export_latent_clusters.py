#!/usr/bin/env python
"""Reference-free latent-space cluster export.

Fits a K-component Gaussian Mixture Model directly in the FULL cryoDRGN latent
space (all zdim dimensions, standardized) and exports one CryoSPARC-importable
.cs particle file per cluster.  This is entirely unsupervised: NO reference map
and NO CryoSPARC class posterior is used to define the clusters.  The GMM sees
only cryoDRGN's latent coordinates.

Why full-dimensional: class separation is often clearer in higher latent
dimensions than in a 2-D PC1/PC2 projection, so the GMM is fit on all zdim
dimensions (not the PCA plane).  PCA is used only to (a) order the components
left->right for stable labelling and (b) report how much variance the 2-D view
captures.

Latent rows are matched to the passthrough by ROW ORDER (valid for a fullset
run with no --ind: z row i <-> passthrough row i).  For an --ind subset run,
pass --ind-keep to map z rows back to the original particle order.

Run with the cryodrgn-py310 env from repo root::

    python scripts/cryodrgn/export_latent_clusters.py \\
      --z results_cryodrgn/J264/fullset_D256_z10_ep50/z.49.pkl \\
      --passthrough-cs data/J264/cryosparc_P7_J264_passthrough_particles_all_classes_blob.cs \\
      -k 6 --dataset J264 \\
      -o results_cryodrgn/J264/fullset_D256_z10_ep50/cluster_exports_k6
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
_REPO = os.path.dirname(_SCRIPTS)
for _p in (_REPO, _SCRIPTS, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def load_latent(path):
    with open(path, "rb") as fh:
        z = pickle.load(fh)
    return np.asarray(z, dtype=np.float64)


def min_separation_sd(gmm):
    """Minimum pairwise Mahalanobis-style separation between components, in SD.

    For each pair, distance between means divided by the average spread along
    the connecting axis.  >2 SD => genuinely distinct; <2 SD => overlapping.
    """
    K = gmm.n_components
    means = gmm.means_
    seps = []
    for i in range(K):
        for j in range(i + 1, K):
            d = means[i] - means[j]
            dist = np.linalg.norm(d)
            u = d / (dist + 1e-12)
            si = np.sqrt(u @ gmm.covariances_[i] @ u)
            sj = np.sqrt(u @ gmm.covariances_[j] @ u)
            seps.append(dist / (0.5 * (si + sj) + 1e-12))
    return float(np.min(seps)), float(np.mean(seps))


def _save_cs(path, arr):
    with open(path, "wb") as fh:
        np.save(fh, arr)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--z", required=True, help="cryoDRGN z.<N>.pkl latent file")
    ap.add_argument("--passthrough-cs", required=True,
                    help="importable passthrough .cs (blob [+ctf]) for all particles")
    ap.add_argument("-k", type=int, default=6, help="number of GMM components")
    ap.add_argument("--ind-keep", default=None,
                    help="ind_keep.pkl if the z run used --ind (maps z rows to "
                         "original particle order); omit for a fullset run")
    ap.add_argument("--dataset", default=None,
                    help="label used in filenames/plots (e.g. J264)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-resp", type=float, default=0.0,
                    help="optional: also export a confident subset with GMM "
                         "max-responsibility >= this (0 = disabled)")
    ap.add_argument("-o", "--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    dset = args.dataset or "dataset"

    # 1) load latent, standardize, fit GMM in FULL latent space (reference-free)
    z = load_latent(args.z)
    n, zdim = z.shape
    print(f"[load] latent {z.shape} from {args.z}")
    scaler = StandardScaler().fit(z)
    Xs = scaler.transform(z)

    gmm = GaussianMixture(args.k, covariance_type="full", reg_covar=1e-6,
                          max_iter=500, n_init=10, random_state=args.seed).fit(Xs)
    resp = gmm.predict_proba(Xs)
    hard = resp.argmax(1)
    maxresp = resp.max(1)

    # 2) PCA only for ordering components + reporting (NOT for clustering)
    pca = PCA(n_components=min(3, zdim), random_state=args.seed).fit(Xs)
    evr = pca.explained_variance_ratio_
    comp_pc1 = pca.transform(gmm.means_)[:, 0]
    order = np.argsort(comp_pc1)               # left->right along PC1
    relabel = np.empty(args.k, dtype=int)
    relabel[order] = np.arange(args.k)         # old comp -> new ordered label
    hard_ordered = relabel[hard]
    names = [f"c{n_}" for n_ in range(args.k)]

    min_sep, mean_sep = min_separation_sd(gmm)
    print(f"[pca]  PC1/PC2 explain {evr[0]*100:.1f}% / "
          f"{(evr[1]*100 if len(evr) > 1 else 0):.1f}% of latent variance")
    print(f"[gmm]  K={args.k}  min separation {min_sep:.2f} SD, "
          f"mean {mean_sep:.2f} SD  (>2 = distinct, <2 = overlapping)")
    print(f"[gmm]  cluster populations: "
          + "  ".join(f"{names[i]}={np.mean(hard_ordered == i)*100:.1f}%"
                      for i in range(args.k)))

    # 3) map z rows -> original particle order
    if args.ind_keep:
        with open(args.ind_keep, "rb") as fh:
            ind_keep = np.asarray(pickle.load(fh)).astype(np.int64)
        if len(ind_keep) != n:
            print(f"  WARNING: ind_keep has {len(ind_keep)} entries but z has {n}")
        orig_row = ind_keep
    else:
        orig_row = np.arange(n)

    # 4) load passthrough, subset per cluster, export .cs
    cs = np.load(args.passthrough_cs)
    if len(cs) < int(orig_row.max()) + 1:
        raise SystemExit(f"passthrough has {len(cs)} rows but z maps up to row "
                         f"{int(orig_row.max())}; wrong passthrough file?")
    uids = cs["uid"].astype(np.uint64)

    # sidecar: uid -> cluster + responsibilities (for CryoSPARC uid-subsetting)
    import csv as _csv
    with open(os.path.join(args.outdir, f"{dset}_latent_clusters.csv"), "w",
              newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["uid", "cluster", "max_resp"]
                   + [f"resp_{names[i]}" for i in range(args.k)])
        resp_ordered = resp[:, order]
        for zi in range(n):
            r = int(orig_row[zi])
            w.writerow([int(uids[r]), names[hard_ordered[zi]],
                        f"{maxresp[zi]:.4f}"]
                       + [f"{resp_ordered[zi, i]:.4f}" for i in range(args.k)])

    print(f"\n{'cluster':<8} {'full':>10} {'conf(>='+str(args.min_resp)+')':>14}")
    print("-" * 34)
    for i in range(args.k):
        sel = (hard_ordered == i)
        rows_full = orig_row[sel]
        sub = cs[rows_full].copy()
        _save_cs(os.path.join(args.outdir, f"{dset}_cluster_{names[i]}.cs"), sub)
        n_conf = 0
        if args.min_resp > 0:
            selc = sel & (maxresp >= args.min_resp)
            rows_c = orig_row[selc]
            subc = cs[rows_c].copy()
            _save_cs(os.path.join(args.outdir,
                                  f"{dset}_cluster_{names[i]}_conf.cs"), subc)
            n_conf = len(rows_c)
        print(f"{names[i]:<8} {len(rows_full):>10,} {n_conf:>14,}")

    # 5) diagnostic figure: PC1-PC2 coloured by cluster
    scores = pca.transform(Xs)
    fig, ax = plt.subplots(figsize=(8, 7))
    palette = plt.cm.tab10(np.linspace(0, 1, max(args.k, 3)))
    for i in range(args.k):
        m = hard_ordered == i
        ax.scatter(scores[m, 0], scores[m, 1], s=3, alpha=0.3,
                   color=palette[i], label=f"{names[i]} ({m.mean()*100:.0f}%)",
                   rasterized=True)
    ax.set_xlabel(f"PC1 ({evr[0]*100:.0f}%)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.0f}%)" if len(evr) > 1 else "PC2")
    ax.set_title(f"{dset}: reference-free K={args.k} latent clusters (fit in "
                 f"full {zdim}-D)\nmin sep {min_sep:.2f} SD | mean {mean_sep:.2f} SD",
                 fontsize=11)
    lg = ax.legend(markerscale=6, fontsize=9)
    for h in lg.legend_handles:
        h.set_alpha(1)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, f"{dset}_latent_clusters_k{args.k}.png"),
                dpi=150)
    plt.close(fig)

    print(f"\n[export] wrote {args.k} cluster .cs files + sidecar CSV to {args.outdir}")
    print("[import] CryoSPARC > Import Particle Stack on each .cs; then "
          "ab-initio -> NU-refine.  (blob+CTF present; poses re-derived by "
          "ab-initio.)  Or subset the parent job by uid using the sidecar CSV.")


if __name__ == "__main__":
    main()
