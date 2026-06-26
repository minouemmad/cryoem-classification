#!/usr/bin/env python
"""Free-energy landscape of the cryoDRGN latent along PC1 -- the honest test of
"discrete metastable states vs. one continuous reaction coordinate".

Motivation
----------
CryoSPARC hetero refinement assumes the particles fall into K discrete classes
and reports a (sharpened) posterior per particle.  cryoDRGN instead embeds every
particle as a continuous latent z.  The scientific question for this project is:

    Are P6/P7/P8(/P9/P10) genuine metastable states, or arbitrary slices through
    ONE continuous reaction coordinate?

A GMM will *always* return K components, so it cannot answer this -- it imposes
discreteness.  The model-free way to look is the empirical free energy along the
dominant latent axis (PC1).  Treating the latent density p(z) as a Boltzmann
distribution, the effective free energy (in units of kT) is

        F(PC1) = -log p(PC1)         (shifted so min F = 0)

Then:
  * separate wells with barriers >~ 1 kT  =>  metastable states  (\\/  \\/  \\/)
  * a single well / barriers << kT        =>  one continuous coord (--------)

This script computes p(PC1) by KDE, the free-energy profile, finds the wells and
the barriers between them (in kT), bootstraps the profile to show it is robust,
and overlays the CryoSPARC class membership so you can see whether the discrete
classes line up with real wells or just tile a single basin.  It runs the same
analysis for several datasets (J1442 3-class and J1497 5-class) and draws a
side-by-side comparison.

Run with the cryoDRGN env from repo root::

    python scripts/cryodrgn/cryodrgn_free_energy.py \
      --dataset "J1442:results_cryodrgn/J1442_real/train_z10/z.100.pkl:data/cryosparc_P25_J1442_passthrough_particles_all_classes.cs:data/cryosparc_P25_J1442_00000_particles.cs:6,7,8" \
      --dataset "J1497:results_cryodrgn/J1497_real/train/z.100.pkl:data/gP25W6J1497_passthrough_particles_all_classes.cs:data/cryosparc_P25_J1497_00000_particles.cs:6,7,8,9,10" \
      --n-dummies 6 -o results_cryodrgn/free_energy
"""
from __future__ import annotations

import argparse
import json
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
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import cryodrgn_latent_gmm as clg


# --------------------------------------------------------------------------- #
# Free-energy helpers
# --------------------------------------------------------------------------- #
def free_energy_1d(x, grid, bw_scale=1.0):
    """F(grid) = -log p(grid) in kT, shifted so min = 0.  p via Gaussian KDE."""
    kde = gaussian_kde(x, bw_method="scott")
    kde.set_bandwidth(kde.factor * bw_scale)
    p = kde(grid)
    p = np.clip(p, 1e-300, None)
    F = -np.log(p)
    F -= F.min()
    return F, p


def find_extrema(F):
    """Indices of interior local minima and maxima of a 1-D array."""
    dF = np.diff(F)
    sign = np.sign(dF)
    # carry previous nonzero sign across flats
    for i in range(1, len(sign)):
        if sign[i] == 0:
            sign[i] = sign[i - 1]
    change = np.diff(sign)
    minima = np.where(change > 0)[0] + 1   # - then + => valley
    maxima = np.where(change < 0)[0] + 1   # + then - => peak
    return minima, maxima


def basin_analysis(F, grid, barrier_kt=0.5):
    """Find wells and merge any separated by a barrier (from the shallower side)
    smaller than ``barrier_kt``.  Returns (kept_minima_idx, barriers) where
    barriers is a list of dicts for the surviving interior barriers."""
    minima, maxima = find_extrema(F)
    if len(minima) == 0:
        minima = np.array([int(np.argmin(F))])
    minima = list(minima)

    def barrier_between(i_left, i_right):
        seg = F[i_left:i_right + 1]
        k = i_left + int(np.argmax(seg))
        top = F[k]
        # height seen from the *shallower* of the two wells = the real separation
        side = top - max(F[i_left], F[i_right])
        return k, top, side

    # iteratively merge the weakest barrier until all survivors exceed threshold
    while len(minima) > 1:
        bars = [barrier_between(minima[j], minima[j + 1])
                for j in range(len(minima) - 1)]
        sides = [b[2] for b in bars]
        jmin = int(np.argmin(sides))
        if sides[jmin] >= barrier_kt:
            break
        # merge: drop the shallower of the two wells flanking the weakest barrier
        a, b = minima[jmin], minima[jmin + 1]
        drop = jmin if F[a] > F[b] else jmin + 1
        minima.pop(drop)

    barriers = []
    for j in range(len(minima) - 1):
        k, top, side = barrier_between(minima[j], minima[j + 1])
        barriers.append({
            "pos": float(grid[k]),
            "F_barrier_kt": float(top),
            "depth_from_shallower_well_kt": float(side),
            "left_well_pos": float(grid[minima[j]]),
            "right_well_pos": float(grid[minima[j + 1]]),
        })
    return np.array(minima), barriers


def bootstrap_band(x, grid, n_boot=200, sub=40000, bw_scale=1.0, seed=0):
    """Bootstrap F(PC1) to show the landscape is not a sampling artefact."""
    rng = np.random.default_rng(seed)
    n = len(x)
    take = min(sub, n)
    Fs = np.empty((n_boot, len(grid)))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=take)
        Fs[b], _ = free_energy_1d(x[idx], grid, bw_scale=bw_scale)
    lo, hi = np.percentile(Fs, [2.5, 97.5], axis=0)
    return lo, hi


# --------------------------------------------------------------------------- #
def analyse_dataset(label, z_path, pass_cs, cs, protein_idx, n_dummies,
                    bw_scale, n_boot, seed):
    print(f"\n=== {label} ===")
    z = clg.load_latent(z_path)
    z_a, cryo_post, cryo_hard, uid_a, n_prot = clg.align_z_to_posteriors(
        z, pass_cs, cs, n_dummies, protein_idx)

    Xs = StandardScaler().fit_transform(z_a)
    pca = PCA(n_components=2, random_state=seed).fit(Xs)
    scores = pca.transform(Xs)
    evr = pca.explained_variance_ratio_
    pc1 = scores[:, 0]

    # orient PC1 so CryoSPARC class index increases left->right (cosmetic only)
    class_means = [pc1[cryo_hard == j].mean() for j in range(len(protein_idx))]
    if np.polyfit(range(len(class_means)), class_means, 1)[0] < 0:
        pc1 = -pc1
        scores[:, 0] = -scores[:, 0]
        class_means = [pc1[cryo_hard == j].mean() for j in range(len(protein_idx))]

    lo, hi = np.percentile(pc1, [0.2, 99.8])
    grid = np.linspace(lo, hi, 500)
    F, p = free_energy_1d(pc1, grid, bw_scale=bw_scale)
    minima, barriers = basin_analysis(F, grid)
    band_lo, band_hi = bootstrap_band(pc1, grid, n_boot=n_boot, bw_scale=bw_scale,
                                      seed=seed)

    n_states = len(minima)
    max_barrier = max((b["depth_from_shallower_well_kt"] for b in barriers),
                      default=0.0)
    verdict = ("CONTINUOUS (single basin)" if n_states == 1
               else f"{n_states} METASTABLE BASINS (max barrier "
                    f"{max_barrier:.2f} kT)")
    print(f"[{label}] PC1 var {evr[0]*100:.1f}% | wells={n_states} | "
          f"barriers(kT)={[round(b['depth_from_shallower_well_kt'],2) for b in barriers]}"
          f" | {verdict}")

    return {
        "label": label,
        "n_particles": int(len(pc1)),
        "zdim": int(z_a.shape[1]),
        "protein_idx": list(protein_idx),
        "pc1_explained_var": float(evr[0]),
        "pc2_explained_var": float(evr[1]),
        "n_wells": int(n_states),
        "well_positions": [float(grid[m]) for m in minima],
        "barriers": barriers,
        "max_barrier_kt": float(max_barrier),
        "verdict": verdict,
        # arrays for plotting (not serialised to json)
        "_pc1": pc1, "_scores": scores, "_grid": grid, "_F": F, "_p": p,
        "_band": (band_lo, band_hi), "_minima": minima,
        "_cryo_hard": cryo_hard, "_class_means": class_means,
    }


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_dataset(res, protein_idx, outdir):
    label = res["label"]
    pc1, grid, F, p = res["_pc1"], res["_grid"], res["_F"], res["_p"]
    band_lo, band_hi = res["_band"]
    minima = res["_minima"]
    cryo_hard = res["_cryo_hard"]
    scores = res["_scores"]
    class_names = [f"P{j}" for j in protein_idx]
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(class_names), 3)))

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axH, axF, axS, axK = axes.ravel()

    # (A) PC1 density, stacked by CryoSPARC class
    lo, hi = grid[0], grid[-1]
    bins = np.linspace(lo, hi, 140)
    axH.hist([pc1[cryo_hard == j] for j in range(len(class_names))], bins=bins,
             stacked=True, color=colors[:len(class_names)], label=class_names,
             edgecolor="none")
    axH.plot(grid, p * len(pc1) * (bins[1] - bins[0]), color="black", lw=1.8,
             label="KDE")
    axH.set_xlabel("PC1 (dominant latent axis)")
    axH.set_ylabel("particle count")
    axH.set_title("A. PC1 density coloured by CryoSPARC class")
    axH.legend(fontsize=9)

    # (B) Free energy F(PC1)
    axF.fill_between(grid, band_lo, band_hi, color="steelblue", alpha=0.25,
                     label="95% bootstrap")
    axF.plot(grid, F, color="steelblue", lw=2.4, label="F(PC1) = -log p")
    axF.plot(grid[minima], F[minima], "v", color="crimson", ms=12,
             label="wells")
    for b in res["barriers"]:
        axF.annotate(f"{b['depth_from_shallower_well_kt']:.2f} kT",
                     (b["pos"], b["F_barrier_kt"]), ha="center", va="bottom",
                     fontsize=10, fontweight="bold", color="darkred")
        axF.plot(b["pos"], b["F_barrier_kt"], "^", color="darkred", ms=10)
    axF.axhline(1.0, color="gray", ls=":", lw=1)
    axF.text(grid[0], 1.02, "1 kT", color="gray", fontsize=8, va="bottom")
    axF.set_xlabel("PC1")
    axF.set_ylabel("free energy  F / kT")
    axF.set_title(f"B. Free-energy profile -- {res['verdict']}")
    axF.legend(fontsize=9)

    # (C) 2D scatter PC1 vs PC2 coloured by class
    for j, name in enumerate(class_names):
        m = cryo_hard == j
        axS.scatter(scores[m, 0], scores[m, 1], s=2, alpha=0.2, color=colors[j],
                    label=name, rasterized=True)
    axS.set_xlabel("PC1"); axS.set_ylabel("PC2")
    axS.set_title("C. Latent plane coloured by CryoSPARC class")
    lg = axS.legend(markerscale=6, fontsize=9)
    for h in lg.legend_handles:
        h.set_alpha(1)

    # (D) 2D free-energy contour F(PC1,PC2)
    try:
        xy = np.vstack([scores[:, 0], scores[:, 1]])
        sub = xy[:, np.random.default_rng(0).integers(0, xy.shape[1],
                                                       size=min(40000, xy.shape[1]))]
        kde2 = gaussian_kde(sub)
        gx = np.linspace(scores[:, 0].min(), scores[:, 0].max(), 120)
        gy = np.linspace(scores[:, 1].min(), scores[:, 1].max(), 120)
        GX, GY = np.meshgrid(gx, gy)
        P2 = kde2(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
        F2 = -np.log(np.clip(P2, 1e-300, None))
        F2 -= F2.min()
        F2 = np.clip(F2, 0, 6)
        cf = axK.contourf(GX, GY, F2, levels=20, cmap="viridis_r")
        axK.contour(GX, GY, F2, levels=10, colors="k", linewidths=0.3, alpha=0.4)
        fig.colorbar(cf, ax=axK, fraction=0.046, pad=0.04, label="F / kT")
    except Exception as e:  # pragma: no cover
        axK.text(0.5, 0.5, f"2D KDE failed: {e}", ha="center")
    axK.set_xlabel("PC1"); axK.set_ylabel("PC2")
    axK.set_title("D. 2D free-energy surface  F(PC1,PC2)")

    fig.suptitle(
        f"{label}: cryoDRGN free-energy landscape  |  {res['n_particles']:,} "
        f"particles  |  zdim {res['zdim']}  |  PC1 = {res['pc1_explained_var']*100:.1f}% var",
        fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = os.path.join(outdir, f"free_energy_{label}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


def plot_comparison(results, outdir):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5.2), squeeze=False)
    for ax, res in zip(axes[0], results):
        grid, F = res["_grid"], res["_F"]
        band_lo, band_hi = res["_band"]
        ax.fill_between(grid, band_lo, band_hi, color="steelblue", alpha=0.25)
        ax.plot(grid, F, color="steelblue", lw=2.4)
        ax.plot(grid[res["_minima"]], F[res["_minima"]], "v", color="crimson",
                ms=11)
        for b in res["barriers"]:
            ax.annotate(f"{b['depth_from_shallower_well_kt']:.2f} kT",
                        (b["pos"], b["F_barrier_kt"]), ha="center", va="bottom",
                        fontsize=9, fontweight="bold", color="darkred")
        ax.axhline(1.0, color="gray", ls=":", lw=1)
        ax.set_xlabel("PC1")
        ax.set_ylabel("F / kT")
        ax.set_title(f"{res['label']}  ({len(res['protein_idx'])}-class)\n"
                     f"{res['verdict']}", fontsize=11)
    fig.suptitle("Free-energy profiles along PC1  --  wells (v) and barriers in kT",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(outdir, "free_energy_compare.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


def write_summary(results, outdir):
    lines = ["# cryoDRGN free-energy landscape (PC1)\n",
             "Treating the latent density as a Boltzmann distribution, "
             "`F(PC1) = -log p(PC1)` in units of kT (min shifted to 0). "
             "Wells separated by barriers >~1 kT are metastable states; "
             "barriers << kT mean one continuous reaction coordinate.\n"]
    for res in results:
        lines.append(f"## {res['label']} ({len(res['protein_idx'])}-class, "
                     f"{res['n_particles']:,} particles)\n")
        lines.append(f"- PC1 explains **{res['pc1_explained_var']*100:.1f}%** "
                     f"of standardized latent variance (PC2 "
                     f"{res['pc2_explained_var']*100:.1f}%).")
        lines.append(f"- Free-energy wells: **{res['n_wells']}** at PC1 "
                     f"= {[round(w,2) for w in res['well_positions']]}.")
        if res["barriers"]:
            for b in res["barriers"]:
                lines.append(
                    f"  - barrier at PC1={b['pos']:.2f}: "
                    f"**{b['depth_from_shallower_well_kt']:.2f} kT** above the "
                    f"shallower well (absolute height {b['F_barrier_kt']:.2f} kT).")
        else:
            lines.append("  - no interior barrier above 0.5 kT.")
        lines.append(f"- **Verdict: {res['verdict']}**\n")
    out = os.path.join(outdir, "free_energy_summary.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[summary] {out}")

    js = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    with open(os.path.join(outdir, "free_energy_metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(js, f, indent=2)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", action="append", required=True,
                    help="LABEL:Z_PKL:PASSTHROUGH_CS:CS:PROT_IDX(comma sep). "
                         "Repeat for each dataset.")
    ap.add_argument("--n-dummies", type=int, default=6)
    ap.add_argument("--bw-scale", type=float, default=1.0,
                    help="KDE bandwidth multiplier (Scott's rule x this). "
                         "<1 sharper, >1 smoother.")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    results = []
    for spec in args.dataset:
        label, z_path, pass_cs, cs, idx = spec.split(":")
        protein_idx = [int(t) for t in idx.split(",")]
        res = analyse_dataset(label, z_path, pass_cs, cs, protein_idx,
                              args.n_dummies, args.bw_scale, args.n_boot,
                              args.seed)
        plot_dataset(res, protein_idx, args.outdir)
        results.append(res)

    if len(results) > 1:
        plot_comparison(results, args.outdir)
    write_summary(results, args.outdir)
    print(f"\n[done] free-energy landscape -> {args.outdir}")


if __name__ == "__main__":
    main()
