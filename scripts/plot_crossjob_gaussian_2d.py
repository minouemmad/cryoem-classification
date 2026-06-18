"""Cross-job pairwise posterior scatter (fig4 style): J1069 class assignments
plotted in J1442 soft-posterior space, with empirical Gaussian overlays.

Matches the format of diagnostics/fig4_pairwise_posterior_scatter_J1442.png:
one panel per pair of protein classes, every particle a dot in (P_i, P_j) raw
posterior space, coloured by hard class, with a per-class empirical Gaussian
overlay (mean dot, +/-1 sigma error bars, 1 sigma solid fill ellipse, 2 sigma
dashed ellipse).

The twist (cross-job weighting idea): the COORDINATES come from the honest
J1442 soft posteriors, while the COLOURS / class assignments come from J1069's
(over-confident) hard argmax, matched particle-for-particle by uid. So this
shows "weight the 1069 classes by the 1442 posteriors and look at their 2-D
Gaussians".

beta=1 (raw J1442 posteriors) is the default and matches the ``weighted_b1``
export; pass --beta 2/4 to sharpen the coordinates toward the simplex corners.

Usage
-----
    python scripts/plot_crossjob_gaussian_2d.py \
        --cs data/cryosparc_P25_J1069_00042_particles.cs \
        --weights-cs data/cryosparc_P25_J1442_00000_particles.cs \
        --n-dummies 6 --outdir results_J1069
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gmm_pipeline import load_posteriors


# Colours matching the reference figure: P6 green, P7 red, P8 blue.
_COLORS = ["#2ca02c", "#d62728", "#1f77b4", "#9467bd", "#ff7f0e",
           "#8c564b", "#e377c2"]


def _ellipse_from_cov(mean, cov, n_std=1.0, **kwargs):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2.0 * n_std * np.sqrt(np.maximum(vals, 0))
    return Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)


def _draw_pairwise_panel(ax, post, hard, labels, i, j, subsample,
                         point_size=2.0, point_alpha=0.15, fontsize=10):
    K = post.shape[1]
    for k in range(K):
        mask = hard[subsample] == k
        ax.scatter(post[subsample][mask, i], post[subsample][mask, j],
                   s=point_size, alpha=point_alpha,
                   color=_COLORS[k % len(_COLORS)], rasterized=True)

    for k in range(K):
        mask = hard == k
        if mask.sum() < 5:
            continue
        pts = post[mask][:, [i, j]]
        mu = pts.mean(axis=0)
        cov = np.atleast_2d(np.cov(pts, rowvar=False))
        color = _COLORS[k % len(_COLORS)]
        r, g, b, _ = mcolors.to_rgba(color)
        dark = (r * 0.6, g * 0.6, b * 0.6, 1.0)

        ax.add_patch(_ellipse_from_cov(mu, cov, n_std=2.0, edgecolor=dark,
                                       facecolor="none", linewidth=1.6,
                                       linestyle=(0, (5, 3)), alpha=0.9,
                                       zorder=6))
        ax.add_patch(_ellipse_from_cov(mu, cov, n_std=1.0, edgecolor="none",
                                       facecolor=color, alpha=0.28, zorder=7))
        ax.add_patch(_ellipse_from_cov(mu, cov, n_std=1.0, edgecolor=dark,
                                       facecolor="none", linewidth=2.0,
                                       alpha=1.0, zorder=8))

        sx, sy = np.sqrt(np.diag(cov))
        ax.errorbar(mu[0], mu[1], xerr=sx, yerr=sy, fmt="none", ecolor=dark,
                    elinewidth=1.1, capsize=2.5, zorder=9)
        ax.plot(mu[0], mu[1], "o", color="white", markeredgecolor=dark,
                markeredgewidth=1.5, markersize=6, zorder=10)

    ax.set_xlabel(f"P({labels[i]})", fontsize=fontsize)
    ax.set_ylabel(f"P({labels[j]})", fontsize=fontsize)
    ax.set_title(f"{labels[i]}  vs  {labels[j]}", fontsize=fontsize)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cs", required=True,
                   help="Particle stack / CLASS-ASSIGNMENT source (J1069). Its "
                        "argmax hard class labels colour the points.")
    p.add_argument("--weights-cs", required=True,
                   help="POSTERIOR-COORDINATE source (J1442). Its soft "
                        "posteriors position the points. Matched to --cs by uid.")
    p.add_argument("--n-dummies", type=int, default=6)
    p.add_argument("--protein-idx", type=int, nargs="*", default=None)
    p.add_argument("--beta", type=float, default=1.0,
                   help="Sharpening exponent on the J1442 posteriors used as "
                        "coordinates (1 = raw / b1; >1 sharpens toward corners).")
    p.add_argument("--max-points", type=int, default=30_000)
    p.add_argument("--outdir", default="results")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.outdir) / "sanity_crossjob"
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading J1069 class assignments from {args.cs}")
    lab_full = load_posteriors(args.cs, protein_idx=args.protein_idx,
                               n_dummies=args.n_dummies)
    labels = [f"P{int(c)}" for c in lab_full.protein_idx]
    lab = lab_full.protein_only()
    lab_class_by_uid = {int(u): int(c) for u, c in zip(lab.uid, lab.hard_class)}
    print(f"      N(J1069 protein)={len(lab.uid):,}  labels={labels}")

    print(f"[2/4] Loading J1442 soft posteriors from {args.weights_cs}")
    coord = load_posteriors(args.weights_cs, protein_idx=args.protein_idx,
                            n_dummies=args.n_dummies).protein_only()
    coord_post_by_uid = {int(u): coord.posterior[i]
                         for i, u in enumerate(coord.uid)}

    print("[3/4] Matching particles by uid (J1069 labels x J1442 coords)")
    matched = [u for u in (int(x) for x in coord.uid) if u in lab_class_by_uid]
    post = np.array([coord_post_by_uid[u] for u in matched])
    hard = np.array([lab_class_by_uid[u] for u in matched])
    if args.beta != 1.0:
        cb = np.power(np.clip(post, 1e-12, None), args.beta)
        post = cb / cb.sum(axis=1, keepdims=True)
    counts = np.bincount(hard, minlength=len(labels))
    print(f"      matched N={len(matched):,}  "
          f"class counts: {dict(zip(labels, counts.tolist()))}")

    print("[4/4] Rendering fig4-style pairwise panels")
    K = post.shape[1]
    pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
    subsample = np.random.default_rng(args.seed).choice(
        len(post), size=min(args.max_points, len(post)), replace=False)

    ncol = max(1, len(pairs))
    fig, axes = plt.subplots(1, ncol, figsize=(5 * ncol, 5), squeeze=False)
    beta_txt = "" if args.beta == 1.0 else f", posterior^{args.beta:g}"
    fig.suptitle(
        f"Pairwise posterior scatter \u2014 J1069 hard classes in J1442 "
        f"posterior space  [K={K}{beta_txt}]\n"
        "Dots = particles, colored by J1069 (CryoSPARC) hard assignment.   "
        "Ellipses = 1\u03c3 (solid) and 2\u03c3 (dashed) per class.\n"
        "White dot = per-class mean,  error bars = \u00b11\u03c3 in each axis.",
        fontsize=11, y=1.04)
    for ax, (i, j) in zip(axes.ravel(), pairs):
        _draw_pairwise_panel(ax, post, hard, labels, i, j, subsample)

    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=_COLORS[k % len(_COLORS)],
                          markeredgecolor="black", markersize=8)
               for k in range(K)]
    fig.legend(handles, labels, fontsize=9, loc="lower center", ncol=K,
               bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()

    suffix = "" if args.beta == 1.0 else f"_b{args.beta:g}"
    fname = out / f"pairwise_J1069class_in_J1442post{suffix}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      saved {fname}")
    print("done.")


if __name__ == "__main__":
    main()
