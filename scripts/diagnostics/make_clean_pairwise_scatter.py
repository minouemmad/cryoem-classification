"""Regenerate the pairwise posterior scatter (fig4) without "Figure 4" in the
title and with proper biological class labels.

Usage (from repo root, no PYTHONPATH needed)::

    python scripts/diagnostics/make_clean_pairwise_scatter.py \
        --cs data/cryosparc_P25_J1442_00000_particles.cs \
        --n-dummies 6 --protein-idx 6 7 8 \
        --dataset J1442 \
        -o results_cryosparc/diagnostics
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent
_REPO = _SCRIPTS.parent
_CRYODRGN_SCRIPTS = _SCRIPTS / "cryodrgn"
for _p in (_REPO, _SCRIPTS, _CRYODRGN_SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

import class_names as cnames
from gmm_pipeline.data_io import load_posteriors


def draw_covariance_ellipse(ax, mean, cov, color, nsigs=(1, 2), alpha_fill=0.08):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    for ns in sorted(nsigs):
        w, h = 2 * ns * np.sqrt(np.maximum(vals, 1e-12))
        e = Ellipse(mean, w, h, angle=angle, fill=(ns == min(nsigs)),
                    edgecolor=color, facecolor=color, lw=2,
                    alpha=alpha_fill if ns == min(nsigs) else 0,
                    linestyle="--" if ns > 1 else "-", zorder=3)
        ax.add_patch(e)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cs", required=True)
    ap.add_argument("--n-dummies", type=int, default=6)
    ap.add_argument("--protein-idx", type=int, nargs="+", default=[6, 7, 8])
    ap.add_argument("--dataset", default=None)
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--alpha", type=float, default=0.12,
                    help="Point transparency (default 0.12 for dense overlapping clouds)")
    ap.add_argument("--n-sample", type=int, default=50000,
                    help="Subsample for scatter (faster rendering; default 50k)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    dset = args.dataset or cnames.guess_dataset(args.cs, args.outdir)
    prot = load_posteriors(args.cs, args.protein_idx, args.n_dummies).protein_only()
    P = prot.posterior          # (N, K), each row sums to 1
    hard = prot.hard_class      # (N,) 0-indexed into protein classes

    K = len(args.protein_idx)
    bio_labels = cnames.labels_for(dset, args.protein_idx)  # e.g. "P6 (NBD1LessMix-Ablated)"
    short_labels = [f"P{j}" for j in args.protein_idx]
    axis_labels = [f"P({lbl})" for lbl in short_labels]

    # Colour per class (consistent with landscape.py)
    palette = plt.cm.Set1(np.linspace(0, 1, max(K, 3)))
    colors = [palette[k] for k in range(K)]

    rng = np.random.default_rng(0)
    idx_sub = rng.choice(len(P), min(args.n_sample, len(P)), replace=False)

    # All pairwise panels
    pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
    ncols = len(pairs)
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 5.0))
    if ncols == 1:
        axes = [axes]

    for ax, (ci, cj) in zip(axes, pairs):
        xi = P[idx_sub, ci]
        xj = P[idx_sub, cj]
        hi = hard[idx_sub]
        for k in range(K):
            m = hi == k
            ax.scatter(xi[m], xj[m], s=3, alpha=args.alpha,
                       color=colors[k], rasterized=True)

        # GMM-style ellipses per class (empirical mean + cov from ALL particles in class)
        for k in range(K):
            m_all = hard == k
            if m_all.sum() > 10:
                pts = P[m_all][:, [ci, cj]]
                mu = pts.mean(0)
                cov_k = np.cov(pts.T) if pts.shape[0] > 2 else np.eye(2) * 1e-4
                ax.errorbar(*mu, xerr=pts[:, 0].std(), yerr=pts[:, 1].std(),
                            fmt="o", ms=10, color="white", ecolor=colors[k],
                            elinewidth=2.5, capsize=5, zorder=5, mec="black", mew=1)
                draw_covariance_ellipse(ax, mu, cov_k, colors[k])

        ax.set_xlabel(axis_labels[ci], fontsize=13)
        ax.set_ylabel(axis_labels[cj], fontsize=13)
        ax.set_title(f"{short_labels[ci]} vs {short_labels[cj]}", fontsize=13)

        # Equal probability line (all mass in those two → sums to ~1 for the pair)
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.set_xlim(lims); ax.set_ylim(lims)

    # Legend with biological names
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[k],
                      markersize=10, label=bio_labels[k]) for k in range(K)]
    fig.legend(handles=handles, loc="lower center", ncol=K, fontsize=11,
               bbox_to_anchor=(0.5, -0.04), frameon=True)

    # Suptitle WITHOUT "Figure N"
    fig.suptitle(
        f"CryoSPARC class posterior overlap — {dset or 'J1442'}, K={K}\n"
        f"Each dot = one particle, coloured by CryoSPARC hard assignment.  "
        f"Ellipses = 1σ/2σ empirical class distribution.",
        fontsize=12, y=1.02)

    fig.tight_layout()
    out = os.path.join(args.outdir, f"pairwise_posterior_scatter_{dset or 'J1442'}.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
