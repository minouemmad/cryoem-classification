#!/usr/bin/env python
"""Drug-conditioned conformational landscape: ΔF(z) between conditions.

This is the analysis half of the drug-landscape idea. It consumes ONE *shared*
cryoDRGN latent (a single model trained on the POOLED particles of every drug
condition) plus a per-particle condition label, and reports how each drug
reshapes the continuous conformational free-energy surface relative to a
reference condition:

    F(z | condition) = -log p(z | condition)          (per-condition free energy)
    ΔF(z)            = F(z | drug) - F(z | reference)  = -log[ p_drug / p_ref ]

Positive ΔF = the drug DEPLETES that region; negative ΔF = the drug STABILISES
(enriches) it. A bootstrap NULL FLOOR (split the reference condition into two
random halves and compute their ΔF) tells you which ΔF features are real versus
sampling/KDE noise -- essential so you do not over-read wiggles.

IMPORTANT (why a *shared* latent is required): cryoDRGN latent axes are arbitrary
per training run, so you CANNOT overlay independently-trained models. Every
condition must be embedded by the SAME encoder -> train one model on the pooled
stack (see the joint-embedding runbook), then pass its z here with --conditions
in the pooled order.

Condition assignment (--conditions):
  * "LABEL1=N1,LABEL2=N2,..."  sequential blocks in the pooled z order
                               (rows 0..N1-1 = LABEL1, next N2 = LABEL2, ...).
                               This matches how you concatenate the stacks.
  * "random"                   split all rows into two random halves A/B
                               (VALIDATION ONLY: the ΔF should stay within the
                               null band everywhere -> proves no false signal).

Run with the cryoDRGN env from repo root, e.g. (real use, after joint training)::

    python scripts/cryodrgn/cryodrgn_drug_landscape.py \
      --z results_cryodrgn/joint_E1371Q/train/z.50.pkl \
      --conditions "ATP=301770,IDOR4=78509" --ref ATP \
      -o results_cryodrgn/drug_landscape/E1371Q_ATP_vs_IDOR4

Validation (machinery only, single dataset, expect ~flat ΔF)::

    python scripts/cryodrgn/cryodrgn_drug_landscape.py \
      --z results_cryodrgn/J264/pilot_z10/z.50.pkl --conditions random \
      -o results_cryodrgn/drug_landscape/_nulltest
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
_REPO = os.path.dirname(_SCRIPTS)
for p in (_REPO, _SCRIPTS, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import cryodrgn_latent_gmm as clg


# --------------------------------------------------------------------------- #
def assign_conditions(spec, n, seed):
    """Return (labels_array[n], ordered_unique_labels)."""
    if spec.strip() == "random":
        rng = np.random.default_rng(seed)
        lab = np.where(rng.random(n) < 0.5, "A", "B")
        return lab, ["A", "B"]
    blocks = []
    for part in spec.split(","):
        name, cnt = part.split("=")
        blocks.append((name.strip(), int(cnt)))
    total = sum(c for _, c in blocks)
    if total != n:
        raise SystemExit(f"--conditions counts sum to {total} but z has {n} rows")
    lab = np.empty(n, dtype=object)
    i = 0
    for name, c in blocks:
        lab[i:i + c] = name
        i += c
    return lab.astype(str), [b[0] for b in blocks]


def kde_on_grid(xy, grid_xy, bw_scale, sub, seed):
    """Gaussian-KDE density of the 2-D points xy (2,N) at grid_xy (2,G)."""
    rng = np.random.default_rng(seed)
    take = min(sub, xy.shape[1])
    fit = xy[:, rng.integers(0, xy.shape[1], size=take)]
    kde = gaussian_kde(fit, bw_method="scott")
    kde.set_bandwidth(kde.factor * bw_scale)
    p = kde(grid_xy).reshape(-1)
    return np.clip(p, 1e-300, None)


def delta_F(p_cond, p_ref):
    """ΔF = -log(p_cond/p_ref), shifted to median 0 for display."""
    dF = -(np.log(p_cond) - np.log(p_ref))
    return dF - np.median(dF)


# --------------------------------------------------------------------------- #
def analyse(args):
    os.makedirs(args.out, exist_ok=True)
    z = clg.load_latent(args.z)
    n = len(z)
    labels, order = assign_conditions(args.conditions, n, args.seed)
    ref = args.ref or order[0]
    if ref not in order:
        raise SystemExit(f"--ref {ref} not among conditions {order}")
    others = [c for c in order if c != ref]
    print(f"[cond] {n:,} particles | conditions "
          + ", ".join(f"{c}={int((labels==c).sum()):,}" for c in order)
          + f" | ref={ref}")

    # shared PCA(2) plane on the joint latent
    Xs = StandardScaler().fit_transform(z)
    pca = PCA(n_components=2, random_state=args.seed).fit(Xs)
    scores = pca.transform(Xs)
    evr = pca.explained_variance_ratio_

    lo = np.percentile(scores, 0.5, axis=0)
    hi = np.percentile(scores, 99.5, axis=0)
    gx = np.linspace(lo[0], hi[0], args.grid)
    gy = np.linspace(lo[1], hi[1], args.grid)
    GX, GY = np.meshgrid(gx, gy)
    grid_xy = np.vstack([GX.ravel(), GY.ravel()])

    def cond_xy(c):
        m = labels == c
        return scores[m].T, m

    # per-condition density
    dens = {}
    for c in order:
        xy, _ = cond_xy(c)
        dens[c] = kde_on_grid(xy, grid_xy, args.bw_scale, args.sub, args.seed)

    # null floor: split the REF condition into two halves, ΔF(halfA, halfB)
    ref_xy, ref_m = cond_xy(ref)
    rng = np.random.default_rng(args.seed)
    half = rng.random(ref_xy.shape[1]) < 0.5
    pA = kde_on_grid(ref_xy[:, half], grid_xy, args.bw_scale, args.sub, args.seed)
    pB = kde_on_grid(ref_xy[:, ~half], grid_xy, args.bw_scale, args.sub, args.seed + 1)
    # only score where the reference actually has density (ignore -log p tails)
    support = dens[ref] > (args.support_pct / 100.0) * dens[ref].max()
    dF_null = delta_F(pA, pB)
    null_band = float(np.percentile(np.abs(dF_null[support]), 95))
    print(f"[null] 95th-pct |ΔF| between random halves of {ref} = "
          f"{null_band:.3f} kT (features beyond this are real)")

    results = {"ref": ref, "conditions": order,
               "counts": {c: int((labels == c).sum()) for c in order},
               "pc1_var": float(evr[0]), "pc2_var": float(evr[1]),
               "null_band_kt": null_band, "shifts": {}}

    for c in others:
        dF = delta_F(dens[c], dens[ref])
        sig = support & (np.abs(dF) > null_band)
        stab = support & (dF < -null_band)   # drug enriches (lowers F)
        depl = support & (dF > null_band)    # drug depletes (raises F)
        # scalar shift metrics
        pr = dens[ref] / dens[ref].sum()
        pc = dens[c] / dens[c].sum()
        m = pr + pc > 0
        kl = float(np.sum(pc[m] * np.log((pc[m]) / (pr[m] + 1e-300) + 1e-300)))
        mid = 0.5 * (pr + pc)
        js = float(0.5 * np.sum(pc[m] * np.log(pc[m] / mid[m] + 1e-300))
                   + 0.5 * np.sum(pr[m] * np.log(pr[m] / mid[m] + 1e-300)))
        w1 = float(wasserstein_distance(scores[labels == c, 0],
                                        scores[labels == ref, 0]))
        frac_sig = float(sig.sum()) / float(max(support.sum(), 1))
        results["shifts"][c] = {
            "kl_cond_ref": kl, "js": js, "wasserstein_pc1": w1,
            "frac_support_significant": frac_sig,
            "max_stabilise_kt": float(-dF[support].min()),
            "max_deplete_kt": float(dF[support].max())}
        print(f"[shift] {c} vs {ref}: JS={js:.4f}  W1(PC1)={w1:.3f}  "
              f"KL={kl:.4f}  significant area={frac_sig*100:.1f}%  "
              f"(max stabilise {-dF[support].min():.2f} / "
              f"deplete {dF[support].max():.2f} kT)")
        _plot_condition(args, c, ref, scores, labels, dens, gx, gy,
                        dF.reshape(args.grid, args.grid),
                        support.reshape(args.grid, args.grid),
                        null_band, evr)

    _write(args, results)
    _plot_overview(args, order, ref, scores, labels, gx, gy, dens, evr)
    print(f"[done] {args.out}")
    return results


# --------------------------------------------------------------------------- #
def _plot_condition(args, c, ref, scores, labels, dens, gx, gy, dF2, support2,
                    null_band, evr):
    GX, GY = np.meshgrid(gx, gy)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    # ref & cond densities
    for a, name in ((ax[0], ref), (ax[1], c)):
        d = dens[name].reshape(len(gy), len(gx))
        a.contourf(GX, GY, d, levels=20, cmap="viridis")
        m = labels == name
        a.set_title(f"p(z | {name})  (n={int(m.sum()):,})")
        a.set_xlabel(f"PC1 ({evr[0]*100:.0f}%)")
        a.set_ylabel(f"PC2 ({evr[1]*100:.0f}%)")
    # ΔF map, masked to reference support, thresholded at the null band
    dFm = np.where(support2, dF2, np.nan)
    vmax = max(np.nanmax(np.abs(dFm)), null_band * 1.5)
    im = ax[2].contourf(GX, GY, dFm, levels=np.linspace(-vmax, vmax, 21),
                        cmap="RdBu_r", extend="both")
    ax[2].contour(GX, GY, np.where(support2, np.abs(dF2), 0), levels=[null_band],
                  colors="k", linewidths=1.2, linestyles="--")
    fig.colorbar(im, ax=ax[2], fraction=0.046, label="ΔF (kT)  red=depleted, blue=stabilised")
    ax[2].set_title(f"ΔF(z) = F({c}) - F({ref})\n(dashed = null band {null_band:.2f} kT)")
    ax[2].set_xlabel(f"PC1 ({evr[0]*100:.0f}%)")
    ax[2].set_ylabel(f"PC2 ({evr[1]*100:.0f}%)")
    fig.suptitle(f"Drug-induced landscape reshaping: {c} vs {ref}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(args.out, f"deltaF_{c}_vs_{ref}.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[plot] {p}")


def _plot_overview(args, order, ref, scores, labels, gx, gy, dens, evr):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
    palette = plt.cm.tab10(np.linspace(0, 1, max(len(order), 3)))
    for i, c in enumerate(order):
        m = labels == c
        ax[0].scatter(scores[m, 0], scores[m, 1], s=2, alpha=0.15,
                      color=palette[i], label=c, rasterized=True)
    ax[0].set_xlabel(f"PC1 ({evr[0]*100:.0f}%)"); ax[0].set_ylabel(f"PC2 ({evr[1]*100:.0f}%)")
    ax[0].set_title("shared latent, coloured by condition")
    lg = ax[0].legend(markerscale=6, fontsize=9)
    for h in lg.legend_handles:
        h.set_alpha(1)
    for i, c in enumerate(order):
        ax[1].hist(scores[labels == c, 0], bins=120, density=True, histtype="step",
                   lw=2, color=palette[i], label=c)
    ax[1].set_xlabel(f"PC1 ({evr[0]*100:.0f}%)"); ax[1].set_ylabel("density")
    ax[1].set_title("PC1 population shift by condition")
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(args.out, "landscape_overview.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[plot] {p}")


def _write(args, results):
    with open(os.path.join(args.out, "drug_landscape_metrics.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    lines = ["# Drug-conditioned conformational landscape", "",
             f"- reference condition: **{results['ref']}**",
             f"- shared latent PC1/PC2 variance: {results['pc1_var']*100:.1f}% / "
             f"{results['pc2_var']*100:.1f}%",
             f"- null band (95th-pct |ΔF| between random halves of ref): "
             f"**{results['null_band_kt']:.3f} kT** — only ΔF beyond this is real.",
             "", "| drug | JS | W1(PC1) | KL | significant area | max stabilise (kT) "
             "| max deplete (kT) |", "|---|---|---|---|---|---|---|"]
    for c, s in results["shifts"].items():
        lines.append(f"| {c} | {s['js']:.4f} | {s['wasserstein_pc1']:.3f} | "
                     f"{s['kl_cond_ref']:.4f} | "
                     f"{s['frac_support_significant']*100:.1f}% | "
                     f"{s['max_stabilise_kt']:.2f} | {s['max_deplete_kt']:.2f} |")
    lines += ["", "Interpretation: **negative ΔF (blue) = the drug stabilises "
              "(enriches) that conformational region; positive ΔF (red) = the "
              "drug depletes it.** A near-zero 'significant area' vs the null "
              "band means the drug does not measurably reshape the ensemble "
              "(e.g. the paper's VX770-alone result)."]
    with open(os.path.join(args.out, "drug_landscape_summary.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--z", required=True, help="shared/joint cryoDRGN z.*.pkl")
    ap.add_argument("--conditions", required=True,
                    help='"LABEL1=N1,LABEL2=N2,..." (pooled order) or "random"')
    ap.add_argument("--ref", default=None, help="reference condition label")
    ap.add_argument("--grid", type=int, default=120)
    ap.add_argument("--bw-scale", type=float, default=1.0)
    ap.add_argument("--sub", type=int, default=40000, help="KDE fit subsample")
    ap.add_argument("--support-pct", type=float, default=2.0,
                    help="ignore grid cells below this %% of ref peak density")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--out", required=True)
    analyse(ap.parse_args(argv))


if __name__ == "__main__":
    main()
