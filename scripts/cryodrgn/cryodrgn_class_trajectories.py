#!/usr/bin/env python
"""Per-class latent trajectories + density-break test: are CryoSPARC classes
substates of one basin or genuinely separated states?

This is the model-free, decoder-free half of the "do P6 and P10 (or P7/P8/P9)
sit on one continuous trajectory?" question.  For each pair of CryoSPARC classes
we draw the straight line between their latent centroids (in the full
standardized latent space), project every particle of the two classes onto it,
and look at the 1-D free energy F = -log p along that connecting coordinate:

    * a single merged peak / barrier << kT   => smooth interpolation, ONE basin
      (the two classes are substates) :  P6 .... P10  =  \\________/
    * two peaks split by a barrier >~ 1 kT    => a real density break, TWO states
      :  P6 ... P10  =  \\/ \\/

We report, for every class pair:
  * the barrier height (kT) on the centroid-connecting axis (the density break),
  * the 1-D distribution-overlap coefficient (1 = identical, 0 = disjoint),
  * the centroid separation in latent SD units.
We also order the classes along PC1 (the dominant reaction coordinate) to read
off the putative "pathway" (e.g. P6 -> P10 -> P9 -> P7 -> P8) and draw the
per-class PC1 ridgeline so the overlaps are visible.

Class membership uses the CryoSPARC hard assignment (argmax of the protein-only
posterior), validated against the per-class split files the user exported
(data/<JOB>_classes/..._class_0X_..._particles.cs).

Run with the cryoDRGN env from repo root::

    python scripts/cryodrgn/cryodrgn_class_trajectories.py \
      --dataset "J1442:results_cryodrgn/J1442_real/train_z10/z.100.pkl:data/cryosparc_P25_J1442_passthrough_particles_all_classes.cs:data/cryosparc_P25_J1442_00000_particles.cs:6,7,8:data/J1442_classes" \
      --dataset "J1497:results_cryodrgn/J1497_real/train/z.100.pkl:data/gP25W6J1497_passthrough_particles_all_classes.cs:data/cryosparc_P25_J1497_00000_particles.cs:6,7,8,9,10:data/J1497_classes" \
      --n-dummies 6 -o results_cryodrgn/class_trajectories
"""
from __future__ import annotations

import argparse
import glob
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
def load_official_membership(class_dir, protein_idx):
    """uid -> CryoSPARC class index (0..K-1) from the per-class split files."""
    if not class_dir or not os.path.isdir(class_dir):
        return None
    uid_to_class = {}
    for j, p in enumerate(protein_idx):
        hits = glob.glob(os.path.join(class_dir, f"*class_0{p}_*particles.cs"))
        hits = [h for h in hits if "passthrough" not in os.path.basename(h)]
        if not hits:
            hits = glob.glob(os.path.join(class_dir,
                                          f"*passthrough*class_{p}.cs"))
        if not hits:
            print(f"[member] WARNING no per-class file for P{p} in {class_dir}")
            continue
        for u in clg.cs_uids(hits[0]).tolist():
            uid_to_class[int(u)] = j
    return uid_to_class


def free_energy_1d(x, grid, bw_scale=1.0):
    kde = gaussian_kde(x, bw_method="scott")
    kde.set_bandwidth(kde.factor * bw_scale)
    p = np.clip(kde(grid), 1e-300, None)
    F = -np.log(p)
    F -= F.min()
    return F, p


def barrier_on_axis(t_a, t_b, bw_scale=1.0, n_grid=400):
    """Free-energy barrier between two class clouds projected on a 1-D axis.

    Returns dict: barrier_kt (from the shallower well), overlap (1-D overlap
    coefficient), peak positions, and the F profile for plotting.
    """
    pooled = np.concatenate([t_a, t_b])
    lo, hi = np.percentile(pooled, [0.5, 99.5])
    grid = np.linspace(lo, hi, n_grid)
    F, p = free_energy_1d(pooled, grid, bw_scale=bw_scale)

    ma, mb = np.median(t_a), np.median(t_b)
    left, right = sorted([ma, mb])
    win = 0.15 * max(right - left, 1e-9)

    def well(center):
        m = (grid >= center - win) & (grid <= center + win)
        return F[m].min() if m.any() else F[np.argmin(np.abs(grid - center))]

    between = (grid >= left) & (grid <= right)
    if between.sum() >= 2:
        saddle = float(F[between].max())
    else:
        saddle = float(max(well(left), well(right)))
    wa, wb = well(left), well(right)
    barrier = max(saddle - max(wa, wb), 0.0)

    # 1-D overlap coefficient between the two class densities on this axis
    ka = gaussian_kde(t_a, bw_method="scott"); ka.set_bandwidth(ka.factor * bw_scale)
    kb = gaussian_kde(t_b, bw_method="scott"); kb.set_bandwidth(kb.factor * bw_scale)
    pa, pb = ka(grid), kb(grid)
    overlap = float(np.trapz(np.minimum(pa, pb), grid) /
                    max(np.trapz(np.maximum(pa, pb), grid), 1e-12))
    return {"barrier_kt": float(barrier), "overlap": overlap,
            "_grid": grid, "_F": F, "_pa": pa, "_pb": pb,
            "_ma": float(ma), "_mb": float(mb)}


# --------------------------------------------------------------------------- #
def analyse(label, z_path, pass_cs, cs, protein_idx, class_dir, n_dummies,
            bw_scale, seed):
    print(f"\n=== {label} ===")
    z = clg.load_latent(z_path)
    z_a, cryo_post, cryo_hard, uid_a, n_prot = clg.align_z_to_posteriors(
        z, pass_cs, cs, n_dummies, protein_idx)

    # validate argmax membership against the per-class split files
    member = load_official_membership(class_dir, protein_idx)
    official = None
    if member is not None:
        official = np.array([member.get(int(u), -1) for u in uid_a.tolist()])
        ok = official >= 0
        agree = float(np.mean(official[ok] == cryo_hard[ok]))
        print(f"[member] matched {ok.sum():,}/{len(official):,} to per-class "
              f"files; argmax vs official agreement {agree*100:.1f}%")
        # prefer the authoritative split where available
        cryo_hard = np.where(ok, official, cryo_hard)

    Xs = StandardScaler().fit_transform(z_a)
    pca = PCA(n_components=2, random_state=seed).fit(Xs)
    scores = pca.transform(Xs)
    evr = pca.explained_variance_ratio_
    pc1, pc2 = scores[:, 0], scores[:, 1]
    cmeans = [pc1[cryo_hard == j].mean() for j in range(len(protein_idx))]
    if np.polyfit(range(len(cmeans)), cmeans, 1)[0] < 0:
        pc1 = -pc1
        scores[:, 0] = pc1
    names = [f"P{p}" for p in protein_idx]

    # per-class PC1 stats + pathway ordering
    stats = []
    for j, nm in enumerate(names):
        m = cryo_hard == j
        stats.append({"class": nm, "n": int(m.sum()),
                      "pc1_mean": float(pc1[m].mean()),
                      "pc1_median": float(np.median(pc1[m])),
                      "pc1_sd": float(pc1[m].std()),
                      "pc2_mean": float(pc2[m].mean())})
    order = sorted(range(len(names)), key=lambda j: stats[j]["pc1_mean"])
    pathway = " -> ".join(names[j] for j in order)
    print(f"[{label}] PC1 {evr[0]*100:.1f}% PC2 {evr[1]*100:.1f}% | "
          f"pathway (by mean PC1): {pathway}")

    # centroids in full standardized latent space
    cents = np.array([Xs[cryo_hard == j].mean(0) for j in range(len(names))])

    # pairwise density-break on the centroid-connecting axis
    K = len(names)
    barrier = np.full((K, K), np.nan)
    overlap = np.full((K, K), np.nan)
    sep_sd = np.full((K, K), np.nan)
    pair_profiles = {}
    for a in range(K):
        for b in range(a + 1, K):
            u = cents[b] - cents[a]
            nrm = np.linalg.norm(u)
            if nrm < 1e-9:
                continue
            u = u / nrm
            ta = Xs[cryo_hard == a] @ u
            tb = Xs[cryo_hard == b] @ u
            r = barrier_on_axis(ta, tb, bw_scale=bw_scale)
            barrier[a, b] = barrier[b, a] = r["barrier_kt"]
            overlap[a, b] = overlap[b, a] = r["overlap"]
            sep_sd[a, b] = sep_sd[b, a] = float(nrm)   # Xs already unit-var
            pair_profiles[(a, b)] = r
            print(f"    {names[a]}-{names[b]}: barrier {r['barrier_kt']:.2f} kT"
                  f" | overlap {r['overlap']:.2f} | centroid sep {nrm:.2f} SD")

    return {
        "label": label, "names": names, "protein_idx": list(protein_idx),
        "n_particles": int(len(pc1)), "zdim": int(Xs.shape[1]),
        "pc1_explained_var": float(evr[0]), "pc2_explained_var": float(evr[1]),
        "pathway_by_pc1": pathway,
        "class_stats": stats,
        "barrier_kt": np.where(np.isnan(barrier), None, barrier).tolist(),
        "overlap": np.where(np.isnan(overlap), None, overlap).tolist(),
        "centroid_sep_sd": np.where(np.isnan(sep_sd), None, sep_sd).tolist(),
        "_pc1": pc1, "_pc2": pc2, "_cryo_hard": cryo_hard, "_order": order,
        "_barrier": barrier, "_overlap": overlap, "_pairs": pair_profiles,
        "_cents2d": np.array([[stats[j]["pc1_mean"], stats[j]["pc2_mean"]]
                              for j in range(K)]),
    }


# --------------------------------------------------------------------------- #
def plot_dataset(res, outdir):
    label = res["label"]
    names = res["names"]
    pc1, pc2, hard = res["_pc1"], res["_pc2"], res["_cryo_hard"]
    order = res["_order"]
    K = len(names)
    colors = plt.cm.Set1(np.linspace(0, 1, max(K, 3)))

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axR, axS, axB, axP = axes.ravel()

    # (A) PC1 ridgeline ordered by mean PC1 = the pathway
    grid = np.linspace(np.percentile(pc1, 0.3), np.percentile(pc1, 99.7), 400)
    for row, j in enumerate(order):
        x = pc1[hard == j]
        kde = gaussian_kde(x, bw_method="scott")
        d = kde(grid); d = d / d.max() * 0.9
        axR.fill_between(grid, row, row + d, color=colors[j], alpha=0.7)
        axR.plot(grid, row + d, color="k", lw=0.6)
        axR.text(grid[0], row + 0.1, names[j], fontsize=11, fontweight="bold")
    axR.set_yticks([])
    axR.set_xlabel("PC1 (dominant reaction coordinate)")
    axR.set_title(f"A. Per-class PC1 ridgeline -- pathway: {res['pathway_by_pc1']}")

    # (B) latent plane: centroids + pathway polyline
    for j, nm in enumerate(names):
        m = hard == j
        axS.scatter(pc1[m], pc2[m], s=2, alpha=0.12, color=colors[j],
                    rasterized=True)
    c2 = res["_cents2d"]
    axS.plot(c2[order, 0], c2[order, 1], "-o", color="black", lw=2, ms=9,
             zorder=5)
    for j, nm in enumerate(names):
        axS.annotate(nm, c2[j], fontsize=12, fontweight="bold", color="black",
                     ha="center", va="bottom", xytext=(0, 8),
                     textcoords="offset points", zorder=6)
    axS.set_xlabel("PC1"); axS.set_ylabel("PC2")
    axS.set_title("B. Class centroids + PC1-ordered pathway")

    # (C) barrier matrix (density break, kT)
    B = res["_barrier"].copy()
    np.fill_diagonal(B, 0.0)
    im = axB.imshow(B, cmap="magma_r", vmin=0,
                    vmax=max(np.nanmax(B), 1.0))
    axB.set_xticks(range(K)); axB.set_xticklabels(names)
    axB.set_yticks(range(K)); axB.set_yticklabels(names)
    for a in range(K):
        for b in range(K):
            if a != b and not np.isnan(B[a, b]):
                axB.text(b, a, f"{B[a,b]:.2f}", ha="center", va="center",
                         color="white" if B[a, b] > np.nanmax(B) * 0.5
                         else "black", fontsize=10)
    fig.colorbar(im, ax=axB, fraction=0.046, pad=0.04, label="barrier / kT")
    axB.set_title("C. Density-break barrier on centroid axis (kT)\n"
                  "<0.5 = same basin, >1 = real split")

    # (D) F-along-axis for the most-merged pairs (lowest barrier, high overlap)
    pairs = sorted(res["_pairs"].items(),
                   key=lambda kv: kv[1]["barrier_kt"])[:3]
    for (a, b), r in pairs:
        axP.plot(r["_grid"], r["_F"],
                 label=f"{names[a]}-{names[b]}  ({r['barrier_kt']:.2f} kT)",
                 lw=2)
    axP.axhline(1.0, color="gray", ls=":", lw=1)
    axP.text(axP.get_xlim()[0], 1.02, "1 kT", color="gray", fontsize=8)
    axP.set_xlabel("centroid-connecting coordinate")
    axP.set_ylabel("F / kT")
    axP.set_title("D. F along the connecting axis (3 most-merged pairs)")
    axP.legend(fontsize=9)

    fig.suptitle(
        f"{label}: are CryoSPARC classes substates or distinct states?  |  "
        f"{res['n_particles']:,} particles  |  PC1 {res['pc1_explained_var']*100:.1f}%",
        fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(outdir, f"class_trajectories_{label}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


def write_outputs(results, outdir):
    lines = ["# cryoDRGN per-class trajectories + density-break test\n",
             "For each CryoSPARC class pair we project both clouds onto the line "
             "joining their latent centroids and measure the free-energy barrier "
             "`F=-log p` between them. **barrier < ~0.5 kT** = smooth "
             "interpolation / one basin (substates); **barrier > ~1 kT** = a real "
             "density break (distinct states). `overlap` is the 1-D distribution "
             "overlap coefficient (1 = identical).\n"]
    for res in results:
        names = res["names"]
        K = len(names)
        lines.append(f"## {res['label']} ({K}-class, {res['n_particles']:,} "
                     f"particles)\n")
        lines.append(f"- PC1 {res['pc1_explained_var']*100:.1f}% / PC2 "
                     f"{res['pc2_explained_var']*100:.1f}% latent variance.")
        lines.append(f"- **Pathway (by mean PC1): {res['pathway_by_pc1']}**.")
        lines.append("- per-class PC1 mean (n): " + ", ".join(
            f"{s['class']} {s['pc1_mean']:+.2f} ({s['n']:,})"
            for s in res["class_stats"]) + ".")
        lines.append("\n  Density-break barrier (kT), upper triangle:\n")
        lines.append("  | | " + " | ".join(names) + " |")
        lines.append("  |" + "---|" * (K + 1))
        B = res["_barrier"]
        for a in range(K):
            row = [names[a]]
            for b in range(K):
                if a == b:
                    row.append("-")
                elif np.isnan(B[a, b]):
                    row.append("")
                else:
                    row.append(f"{B[a,b]:.2f}")
            lines.append("  | " + " | ".join(row) + " |")
        # call out the substate groupings
        merged = [(names[a], names[b], B[a, b])
                  for a in range(K) for b in range(a + 1, K)
                  if not np.isnan(B[a, b]) and B[a, b] < 0.5]
        split = [(names[a], names[b], B[a, b])
                 for a in range(K) for b in range(a + 1, K)
                 if not np.isnan(B[a, b]) and B[a, b] >= 1.0]
        if merged:
            lines.append("\n  - **No density break (<0.5 kT, same basin):** " +
                         ", ".join(f"{a}-{b} ({v:.2f})" for a, b, v in merged) +
                         ".")
        if split:
            lines.append("  - **Real density break (>=1 kT):** " +
                         ", ".join(f"{a}-{b} ({v:.2f})" for a, b, v in split) +
                         ".")
        lines.append("")
    out = os.path.join(outdir, "class_trajectories_summary.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[summary] {out}")

    js = [{k: v for k, v in r.items() if not k.startswith("_")}
          for r in results]
    with open(os.path.join(outdir, "class_trajectories_metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(js, f, indent=2)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", action="append", required=True,
                    help="LABEL:Z_PKL:PASSTHROUGH_CS:CS:PROT_IDX:CLASS_DIR. "
                         "CLASS_DIR (per-class split files) is optional; leave "
                         "trailing colon empty to skip.")
    ap.add_argument("--n-dummies", type=int, default=6)
    ap.add_argument("--bw-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    results = []
    for spec in args.dataset:
        parts = spec.split(":")
        label, z_path, pass_cs, cs, idx = parts[:5]
        class_dir = parts[5] if len(parts) > 5 and parts[5] else None
        protein_idx = [int(t) for t in idx.split(",")]
        res = analyse(label, z_path, pass_cs, cs, protein_idx, class_dir,
                      args.n_dummies, args.bw_scale, args.seed)
        plot_dataset(res, args.outdir)
        results.append(res)

    write_outputs(results, args.outdir)
    print(f"\n[done] class trajectories + density break -> {args.outdir}")


if __name__ == "__main__":
    main()
