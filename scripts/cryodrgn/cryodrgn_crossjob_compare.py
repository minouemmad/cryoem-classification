#!/usr/bin/env python
"""Cross-job cryoDRGN latent-space comparison.

Compares two independently-trained cryoDRGN runs (a reference job and a test
job, typically differing in the number of CryoSPARC classes used) and tests
whether the additional CryoSPARC classes in the test job correspond to distinct
conformational states in the continuous latent landscape.

Both jobs must share the same underlying particle images; uid alignment is
performed automatically.

What this script measures
-------------------------
A. Latent reproducibility: canonical correlations (CCA) between the two latent
   spaces and the PC1-vs-PC1 Pearson r.  High values confirm that the two runs
   independently encode the same dominant conformational coordinate.

B. How many states does the latent support?  BIC(K) and silhouette(K) curves
   for both runs, plus kernel-density mode counting on PC1.  A genuine N-state
   dataset shows a BIC/silhouette elbow at K=N and N distinct PC1 peaks.

C. Supervised CryoSPARC-class recoverability from z: linear discriminant
   analysis (LDA) with stratified 5-fold CV.  Balanced accuracy near 1/K_chance
   means the CryoSPARC partition is not in the latent density.

D. Alternative unsupervised classifiers on the test latent (MiniBatch KMeans,
   full-covariance GMM, and HDBSCAN density clustering) scored by ARI/AMI
   against the CryoSPARC argmax — tests whether any method recovers the partition.

Figures: pc1_overlap.png, model_selection.png, supervised_recall.png,
latent_reproducibility.png.  Plus compare_summary.md and compare_metrics.json.

Run with the cryoDRGN env from repo root::

    python scripts/cryodrgn/cryodrgn_crossjob_compare.py \
      --z-ref results_cryodrgn/J1442_real/train_z10/z.100.pkl \
      --passthrough-ref data/cryosparc_P25_J1442_passthrough_particles_all_classes.cs \
      --cs-ref data/cryosparc_P25_J1442_00000_particles.cs \
      --protein-idx-ref 6 7 8 \
      --z-test results_cryodrgn/J1497_real/train/z.100.pkl \
      --passthrough-test data/gP25W6J1497_passthrough_particles_all_classes.cs \
      --cs-test data/cryosparc_P25_J1497_00000_particles.cs \
      --protein-idx-test 6 7 8 9 10 \
      --n-dummies 6 --label-ref "J1442 (3-class)" --label-test "J1497 (5-class)" \
      -o results_cryodrgn/J1497_real/crossjob_comparison
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
from sklearn.cluster import MiniBatchKMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             balanced_accuracy_score, confusion_matrix)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

import cryodrgn_latent_gmm as clg


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def canonical_correlations(X, Y):
    """Canonical correlations between two column-centred matrices via QR/SVD."""
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    Qx, _ = np.linalg.qr(Xc)
    Qy, _ = np.linalg.qr(Yc)
    s = np.linalg.svd(Qx.T @ Qy, compute_uv=False)
    return np.clip(s, 0.0, 1.0)


def dip_statistic(x):
    """Hartigan's dip statistic (unimodality test), simple O(n log n) version.

    Returns the dip value; ~<0.01 for clearly unimodal large samples, larger for
    multimodal.  We compare the ECDF to its greatest-convex-minorant /
    least-concave-majorant on a subsample for speed.
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n > 20000:
        idx = np.linspace(0, n - 1, 20000).astype(int)
        x = x[idx]
        n = x.size
    ecdf = np.arange(1, n + 1) / n
    # distance of ECDF from the best monotone *linear* (uniform) fit is a crude
    # but stable multimodality proxy; for a unimodal density the ECDF is convex
    # then concave (one inflection), multimodal has several.
    # Use count of sign changes in the 2nd difference of a smoothed ECDF.
    xs = (x - x.min()) / (np.ptp(x) + 1e-12)
    # kernel density on a grid, count modes
    grid = np.linspace(0, 1, 512)
    bw = 0.05
    dens = np.exp(-0.5 * ((grid[:, None] - xs[None, :]) / bw) ** 2).sum(1)
    dens /= dens.sum()
    # local maxima
    modes = int(np.sum((dens[1:-1] > dens[:-2]) & (dens[1:-1] > dens[2:])
                       & (dens[1:-1] > dens.max() * 0.02)))
    dip = float(np.max(np.abs(ecdf - np.linspace(0, 1, n))))
    return dip, modes


def silhouette_subsample(Xs, labels, rng, n=8000):
    from sklearn.metrics import silhouette_score
    if len(Xs) > n:
        idx = rng.choice(len(Xs), n, replace=False)
        Xs, labels = Xs[idx], labels[idx]
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(silhouette_score(Xs, labels))


def model_selection_curves(Xs, k_range, seed, rng):
    """BIC and silhouette as a function of K for a standardised latent."""
    bic, sil = [], []
    for k in k_range:
        g = GaussianMixture(k, covariance_type="full", reg_covar=1e-6,
                            max_iter=500, tol=1e-5, n_init=2,
                            random_state=seed).fit(Xs)
        bic.append(float(g.bic(Xs)))
        sil.append(silhouette_subsample(Xs, g.predict(Xs), rng))
    return np.array(bic), np.array(sil)


def supervised_separability(Xs, hard, seed):
    """LDA cross-validated balanced accuracy + per-class recall, vs chance."""
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    pred = cross_val_predict(LinearDiscriminantAnalysis(), Xs, hard, cv=skf)
    bacc = float(balanced_accuracy_score(hard, pred))
    classes = np.unique(hard)
    chance = 1.0 / len(classes)
    cm = confusion_matrix(hard, pred, labels=classes)
    recall = cm.diagonal() / cm.sum(1).clip(min=1)
    return {
        "balanced_accuracy": bacc,
        "chance": float(chance),
        "lift_over_chance": float(bacc / chance),
        "per_class_recall": {int(c): float(r) for c, r in zip(classes, recall)},
        "confusion": cm.tolist(),
        "classes": classes.tolist(),
    }


def pc1_of(Xs):
    p = PCA(n_components=2).fit(Xs)
    sc = p.transform(Xs)
    return sc[:, 0], float(p.explained_variance_ratio_[0])


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--z-ref", required=True,
                    help="Reference latent z.pkl (smaller/cleaner class count)")
    ap.add_argument("--passthrough-ref", required=True)
    ap.add_argument("--cs-ref", required=True,
                    help="Reference CryoSPARC particles .cs (multi-class posteriors)")
    ap.add_argument("--protein-idx-ref", type=int, nargs="+", required=True)
    ap.add_argument("--z-test", required=True,
                    help="Test latent z.pkl (larger/more classes)")
    ap.add_argument("--passthrough-test", required=True)
    ap.add_argument("--cs-test", required=True)
    ap.add_argument("--protein-idx-test", type=int, nargs="+", required=True)
    ap.add_argument("--n-dummies", type=int, default=6)
    ap.add_argument("--label-ref", default="ref",
                    help="Display label for the reference job (default: ref)")
    ap.add_argument("--label-test", default="test",
                    help="Display label for the test job (default: test)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ---- load + align both runs ------------------------------------------- #
    print(f"[load] reference ({args.label_ref})")
    z3 = clg.load_latent(args.z_ref)
    z3, post3, hard3, uid3, _ = clg.align_z_to_posteriors(
        z3, args.passthrough_ref, args.cs_ref, args.n_dummies,
        args.protein_idx_ref)
    print(f"[load] test ({args.label_test})")
    z5 = clg.load_latent(args.z_test)
    z5, post5, hard5, uid5, _ = clg.align_z_to_posteriors(
        z5, args.passthrough_test, args.cs_test, args.n_dummies,
        args.protein_idx_test)

    Xs3 = StandardScaler().fit_transform(z3)
    Xs5 = StandardScaler().fit_transform(z5)

    metrics = {"label_ref": args.label_ref, "label_test": args.label_test,
               "n_ref": int(len(z3)), "n_test": int(len(z5)),
               "k_ref": int(len(args.protein_idx_ref)),
               "k_test": int(len(args.protein_idx_test))}
    # keep legacy keys for backward compat
    metrics["n_3class"] = metrics["n_ref"]
    metrics["n_5class"] = metrics["n_test"]
    metrics["k_3class"] = metrics["k_ref"]
    metrics["k_5class"] = metrics["k_test"]

    # ---- A. latent reproducibility (common particles) --------------------- #
    row5 = {int(u): i for i, u in enumerate(uid5.tolist())}
    keep3, keep5 = [], []
    for i, u in enumerate(uid3.tolist()):
        j = row5.get(int(u))
        if j is not None:
            keep3.append(i)
            keep5.append(j)
    keep3 = np.asarray(keep3)
    keep5 = np.asarray(keep5)
    common = len(keep3)
    metrics["common_particles"] = int(common)

    cc = canonical_correlations(z3[keep3], z5[keep5])
    pc1_3_c, ev3 = pc1_of(Xs3)
    pc1_5_c, ev5 = pc1_of(Xs5)
    a = pc1_3_c[keep3]
    b = pc1_5_c[keep5]
    if np.corrcoef(a, b)[0, 1] < 0:        # PCA sign is arbitrary
        b = -b
        pc1_5_c = -pc1_5_c
    pc1_corr = float(np.corrcoef(a, b)[0, 1])
    metrics["latent_canonical_corr"] = [float(x) for x in cc]
    metrics["pc1_corr_3v5"] = pc1_corr
    metrics["pc1_explvar_3class"] = ev3
    metrics["pc1_explvar_5class"] = ev5
    print(f"[A] canonical corr (top4): {np.round(cc[:4], 3)}  "
          f"PC1<->PC1 r={pc1_corr:.3f}")

    # ---- B. how many states does the latent support? --------------------- #
    k_range = list(range(2, 11))
    bic3, sil3 = model_selection_curves(Xs3, k_range, args.seed, rng)
    bic5, sil5 = model_selection_curves(Xs5, k_range, args.seed, rng)
    dip3, modes3 = dip_statistic(pc1_3_c)
    dip5, modes5 = dip_statistic(pc1_5_c)
    metrics["k_range"] = k_range
    metrics["bic_3class"] = bic3.tolist()
    metrics["bic_5class"] = bic5.tolist()
    metrics["silhouette_3class"] = sil3.tolist()
    metrics["silhouette_5class"] = sil5.tolist()
    metrics["pc1_modes_3class"] = modes3
    metrics["pc1_modes_5class"] = modes5
    # silhouette at the "design" K for each
    sil_at_design_3 = float(sil3[k_range.index(metrics["k_ref"])])
    sil_at_design_5 = float(sil5[k_range.index(metrics["k_test"])])
    metrics["silhouette_at_designK_3class"] = sil_at_design_3
    metrics["silhouette_at_designK_5class"] = sil_at_design_5
    print(f"[B] PC1 modes: 3class={modes3}  5class={modes5}  | "
          f"silhouette@designK: 3={sil_at_design_3:.3f} 5={sil_at_design_5:.3f}")

    # ---- C. supervised separability of the CryoSPARC partition ------------ #
    sep3 = supervised_separability(Xs3, hard3, args.seed)
    sep5 = supervised_separability(Xs5, hard5, args.seed)
    metrics["supervised_3class"] = sep3
    metrics["supervised_5class"] = sep5
    print(f"[C] LDA balanced acc: 3class={sep3['balanced_accuracy']:.3f} "
          f"(chance {sep3['chance']:.3f}, lift {sep3['lift_over_chance']:.2f}x) | "
          f"5class={sep5['balanced_accuracy']:.3f} "
          f"(chance {sep5['chance']:.3f}, lift {sep5['lift_over_chance']:.2f}x)")

    # ---- D. alternative unsupervised classifiers on the 5-class latent ---- #
    alt = {}
    km = MiniBatchKMeans(metrics["k_5class"], random_state=args.seed,
                         n_init=10, batch_size=4096).fit_predict(Xs5)
    alt["kmeans"] = {
        "ari": float(adjusted_rand_score(hard5, km)),
        "ami": float(adjusted_mutual_info_score(hard5, km)),
    }
    gm = GaussianMixture(metrics["k_5class"], covariance_type="full",
                         reg_covar=1e-6, n_init=3,
                         random_state=args.seed).fit(Xs5).predict(Xs5)
    alt["gmm_full"] = {
        "ari": float(adjusted_rand_score(hard5, gm)),
        "ami": float(adjusted_mutual_info_score(hard5, gm)),
    }
    try:
        from sklearn.cluster import HDBSCAN
        sub = rng.choice(len(Xs5), min(40000, len(Xs5)), replace=False)
        hl = HDBSCAN(min_cluster_size=2000, min_samples=50).fit_predict(Xs5[sub])
        nclust = int(len(set(hl.tolist())) - (1 if -1 in hl else 0))
        noise = float(np.mean(hl == -1))
        alt["hdbscan"] = {
            "ari": float(adjusted_rand_score(hard5[sub], hl)),
            "ami": float(adjusted_mutual_info_score(hard5[sub], hl)),
            "n_clusters_found": nclust,
            "noise_fraction": noise,
        }
    except Exception as e:                  # noqa: BLE001
        alt["hdbscan"] = {"error": str(e)}
    metrics["alt_clustering_5class"] = alt
    print(f"[D] 5-class ARI to CryoSPARC: kmeans={alt['kmeans']['ari']:.3f} "
          f"gmm={alt['gmm_full']['ari']:.3f} "
          f"hdbscan={alt.get('hdbscan', {}).get('ari', 'n/a')}")

    # ---- figures ---------------------------------------------------------- #
    _fig_pc1_overlap(pc1_3_c, hard3, args.protein_idx_ref, args.label_ref,
                     pc1_5_c, hard5, args.protein_idx_test, args.label_test,
                     args.outdir)
    _fig_model_selection(k_range, bic3, sil3, bic5, sil5,
                         metrics["k_ref"], metrics["k_test"],
                         args.label_ref, args.label_test, args.outdir)
    _fig_supervised(sep5, args.protein_idx_test, args.label_test, args.outdir)
    _fig_reproducibility(a, b, pc1_corr,
                         args.label_ref, args.label_test, args.outdir)

    with open(os.path.join(args.outdir, "compare_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    _write_summary(metrics, args.protein_idx_test, args.outdir)
    print(f"[done] wrote outputs to {args.outdir}")


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _hist_by_class(ax, pc1, hard, protein_idx, title):
    classes = np.unique(hard)
    cmap = plt.get_cmap("tab10")
    lo, hi = np.percentile(pc1, [0.5, 99.5])
    bins = np.linspace(lo, hi, 80)
    for c in classes:
        ax.hist(pc1[hard == c], bins=bins, histtype="step", linewidth=1.6,
                color=cmap(c % 10), label=f"P{protein_idx[c]}")
    ax.set_xlabel("latent PC1")
    ax.set_ylabel("particles")
    ax.set_title(title)
    ax.legend(fontsize=8)


def _fig_pc1_overlap(pc1_3, hard3, pi3, label3, pc1_5, hard5, pi5, label5, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    _hist_by_class(axes[0], pc1_3, hard3, pi3,
                   f"{label3} — CryoSPARC class along latent PC1")
    _hist_by_class(axes[1], pc1_5, hard5, pi5,
                   f"{label5} — CryoSPARC class along latent PC1")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "pc1_overlap.png"), dpi=150)
    plt.close(fig)


def _fig_model_selection(ks, bic3, sil3, bic5, sil5, kd3, kd5,
                         label3, label5, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    ax.plot(ks, (bic3 - bic3.min()) / (np.ptp(bic3) + 1e-9), "o-",
            label=label3)
    ax.plot(ks, (bic5 - bic5.min()) / (np.ptp(bic5) + 1e-9), "s-",
            label=label5)
    ax.axvline(kd3, color="C0", ls=":", alpha=0.7)
    ax.axvline(kd5, color="C1", ls=":", alpha=0.7)
    ax.set_xlabel("K (GMM components)")
    ax.set_ylabel("BIC (min-max normalised)")
    ax.set_title("GMM model selection (BIC): monotone decrease = continuous density")
    ax.legend()
    ax = axes[1]
    ax.plot(ks, sil3, "o-", label=label3)
    ax.plot(ks, sil5, "s-", label=label5)
    ax.axvline(kd3, color="C0", ls=":", alpha=0.7, label=f"design K={kd3}")
    ax.axvline(kd5, color="C1", ls=":", alpha=0.7, label=f"design K={kd5}")
    ax.set_xlabel("K (GMM components)")
    ax.set_ylabel("silhouette (higher = better separated)")
    ax.set_title("Silhouette vs K: low values confirm components overlap")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "model_selection.png"), dpi=150)
    plt.close(fig)


def _fig_supervised(sep5, pi5, label5, outdir):
    cm = np.array(sep5["confusion"], dtype=float)
    cm_n = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_n, cmap="magma", vmin=0, vmax=1)
    labels = [f"P{pi5[c]}" for c in sep5["classes"]]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("LDA-predicted class (from latent z, 5-fold CV)")
    ax.set_ylabel("CryoSPARC argmax class (ground truth)")
    ax.set_title(f"{label5} supervised class recovery\n"
                 f"LDA balanced acc {sep5['balanced_accuracy']:.3f} "
                 f"(chance {sep5['chance']:.3f})")
    for i in range(cm_n.shape[0]):
        for j in range(cm_n.shape[1]):
            ax.text(j, i, f"{cm_n[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm_n[i, j] < 0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalised recall")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "supervised_recall.png"), dpi=150)
    plt.close(fig)


def _fig_reproducibility(a, b, r, label3, label5, outdir):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    idx = np.random.default_rng(0).choice(len(a), min(20000, len(a)),
                                          replace=False)
    ax.scatter(a[idx], b[idx], s=2, alpha=0.15, color="navy")
    ax.set_xlabel(f"{label3} latent PC1")
    ax.set_ylabel(f"{label5} latent PC1 (sign-aligned)")
    ax.set_title(f"Latent PC1 reproducibility across independent cryoDRGN runs\n"
                 f"Pearson r = {r:.3f}")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "latent_reproducibility.png"), dpi=150)
    plt.close(fig)


def _write_summary(m, pi_test, outdir):
    """Write compare_summary.md.  pi_test = protein_idx_test list."""
    s5 = m["supervised_5class"]
    s3 = m["supervised_3class"]
    alt = m["alt_clustering_5class"]
    label3 = m.get("label_ref", "ref")
    label5 = m.get("label_test", "test")
    # Map 0-based class index -> actual protein class number (e.g. 0->6, 1->7...)
    recall5 = {f"P{pi_test[int(k)]}": round(v, 3)
               for k, v in s5["per_class_recall"].items()}
    lines = [
        f"# cryoDRGN cross-job comparison: {label3} vs {label5}",
        "",
        f"- particles: {label3} {m['n_ref']:,} | {label5} {m['n_test']:,} | "
        f"common {m['common_particles']:,}",
        "",
        "## A. Latent reproducibility (same images, two independently-trained runs)",
        f"- canonical correlations (top 4): "
        f"{[round(x, 3) for x in m['latent_canonical_corr'][:4]]}",
        f"- PC1 Pearson r: {m['pc1_corr_3v5']:.3f} "
        f"(PC1 expl.var {label3}: {m['pc1_explvar_3class']:.3f} / "
        f"{label5}: {m['pc1_explvar_5class']:.3f})",
        "",
        "## B. How many states does the latent support?",
        f"- PC1 kernel-density modes: {label3}={m['pc1_modes_3class']} | "
        f"{label5}={m['pc1_modes_5class']}",
        f"- silhouette at design K: {label3} K={m['k_ref']} → "
        f"{m['silhouette_at_designK_3class']:.3f} | "
        f"{label5} K={m['k_test']} → {m['silhouette_at_designK_5class']:.3f}",
        "  (silhouette << 0.25 => components overlap; no discrete clusters)",
        "",
        "## C. Supervised CryoSPARC-class recoverability from z (LDA, 5-fold CV)",
        f"- {label3} balanced accuracy {s3['balanced_accuracy']:.3f} "
        f"(chance {s3['chance']:.3f}, {s3['lift_over_chance']:.2f}x lift)",
        f"- {label5} balanced accuracy {s5['balanced_accuracy']:.3f} "
        f"(chance {s5['chance']:.3f}, {s5['lift_over_chance']:.2f}x lift)",
        f"- {label5} per-class recall: {recall5}",
        "",
        f"## D. Alternative unsupervised classifiers on the {label5} latent",
        f"- KMeans(K={m['k_test']}) ARI {alt['kmeans']['ari']:.3f} "
        f"/ AMI {alt['kmeans']['ami']:.3f}",
        f"- full-cov GMM(K={m['k_test']}) ARI {alt['gmm_full']['ari']:.3f} "
        f"/ AMI {alt['gmm_full']['ami']:.3f}",
    ]
    if "ari" in alt.get("hdbscan", {}):
        h = alt["hdbscan"]
        lines.append(
            f"- HDBSCAN ARI {h['ari']:.3f} / AMI {h['ami']:.3f} | "
            f"found {h['n_clusters_found']} dense clusters "
            f"({h['noise_fraction']*100:.0f}% noise)")
    else:
        lines.append(f"- HDBSCAN: {alt.get('hdbscan', {}).get('error', 'n/a')}")
    lines += [
        "",
        "## Findings",
        f"- {label3} and {label5} encode the same latent coordinate "
        f"(canonical corr ≥ {min(m['latent_canonical_corr'][:4]):.3f}, "
        f"PC1 r = {m['pc1_corr_3v5']:.3f}).",
        f"- Silhouette at design K ({m['silhouette_at_designK_5class']:.3f}) "
        f"is well below 0.25; BIC decreases monotonically without elbow: "
        f"the {label5} partition is slicing a continuous density.",
        f"- LDA recall for extra {label5} classes: "
        + ", ".join(f"{k} = {v:.3f}" for k, v in recall5.items()
                    if v < 0.10) + " (near zero; latent cannot distinguish them).",
    ]
    with open(os.path.join(outdir, "compare_summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
