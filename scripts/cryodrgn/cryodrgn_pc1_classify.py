#!/usr/bin/env python
"""Classify particles along the cryoDRGN PC1 axis and export sets for ab-initio/NU.

John's plan (meeting notes):
  * "Ab-initio - initial res at 12, final resolution at 4 -> do on crude division
     along the pc1 values."                              -> RUN 1 (pc1_crude)
  * "If you could fit a 3 component gaussian, could also try limiting to the most
     confidently assigned particles."                    -> RUN 2 (pc1_gmm / pc1_gmm_conf)
  * "Once you do classification based on pc1, can do particle overlap there too."
                                                          -> overlap_metrics.json + confusion PNGs
  * "There should be 3 different ab-initio-NU runs to do."
       RUN 1 = crude PC1 tertile division
       RUN 2 = 1-D K-component GMM on PC1, confident particles only
       RUN 3 = full latent-space GMM (already exported by export_cryodrgn_subsets.py)

This script reduces the (already converged) cryoDRGN latent to its dominant axis
(PC1) and produces TWO PC1-based partitions plus all the overlap/confusion metrics
that let us judge whether the PC1 split agrees with CryoSPARC's hetero-refinement
classes (the cryoDRGN analogue of CryoSPARC's class-population witness test).

It writes CryoSPARC-importable .cs subsets (subsetted from the passthrough .cs)
for every PC1 class so each can be fed into Ab-initio -> NU-Refinement.

Run with the cryoDRGN env (numpy/sklearn/scipy/matplotlib) from repo root::

    python scripts/cryodrgn/cryodrgn_pc1_classify.py \
      --z results_cryodrgn/J1442_real/train_z10/z.100.pkl \
      --passthrough-cs data/cryosparc_P25_J1442_passthrough_particles_all_classes.cs \
      --cs data/cryosparc_P25_J1442_00000_particles.cs \
      --n-dummies 6 --protein-idx 6 7 8 \
      --gmm-conf 0.8 \
      -o results_cryodrgn/J1442_real/pc1_classify
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             normalized_mutual_info_score)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import cryodrgn_latent_gmm as clg
from gmm_pipeline.confusion import soft_posterior_confusion


# --------------------------------------------------------------------------- #
# Label alignment
# --------------------------------------------------------------------------- #
def align_labels_to_reference(ref_hard, new_hard, k):
    """Permute the integer labels in new_hard so they best match ref_hard.

    Returns relabelled new_hard (same label j means "looks like ref class j").
    """
    cont = np.zeros((k, k), dtype=np.int64)
    for r, n in zip(ref_hard, new_hard):
        if 0 <= r < k and 0 <= n < k:
            cont[r, n] += 1
    rows, cols = linear_sum_assignment(-cont)        # ref-row r <- new-col c
    remap = np.empty(k, dtype=int)
    for r, c in zip(rows, cols):
        remap[c] = r
    return remap[new_hard], remap


def row_normalised_confusion(ref_hard, other_hard, k):
    """C[i,j] = P(other == j | ref == i). Rows sum to 1."""
    C = np.zeros((k, k))
    for i in range(k):
        sel = ref_hard == i
        n = int(sel.sum())
        if n == 0:
            continue
        for j in range(k):
            C[i, j] = float(np.sum(other_hard[sel] == j)) / n
    return C


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def heatmap(M, row_labels, col_labels, title, path, fmt="{:.2f}", vmax=1.0,
            xlabel="", ylabel=""):
    fig, ax = plt.subplots(figsize=(1.6 + 0.95 * M.shape[1],
                                    1.6 + 0.85 * M.shape[0]))
    im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=0, vmax=vmax)
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(col_labels)
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(row_labels)
    thr = 0.6 * vmax
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                    color="white" if M[i, j] < thr else "black", fontsize=9)
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def pc1_hist(pc1, gmm, edges, labels, path):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    lo, hi = np.percentile(pc1, [0.2, 99.8])
    ax.hist(pc1, bins=160, range=(lo, hi), density=True, color="0.82",
            edgecolor="none", label="all particles")
    grid = np.linspace(lo, hi, 500)
    means = gmm.means_.ravel()
    sds = np.sqrt(gmm.covariances_.ravel())
    weights = gmm.weights_.ravel()
    total = np.zeros_like(grid)
    order = np.argsort(means)
    palette = plt.cm.tab10(np.linspace(0, 1, max(len(means), 3)))
    for rank, comp in enumerate(order):
        bell = weights[comp] * np.exp(-0.5 * ((grid - means[comp]) / sds[comp]) ** 2) \
            / (sds[comp] * np.sqrt(2 * np.pi))
        total += bell
        ax.plot(grid, bell, color=palette[rank], lw=2,
                label=f"GMM comp {labels[rank]} (w={weights[comp]:.2f})")
    ax.plot(grid, total, color="black", lw=1.5, ls="--", label="GMM sum")
    for e in edges[1:-1]:
        ax.axvline(e, color="crimson", ls=":", lw=1.3)
    ax.set_xlabel("PC1 (dominant cryoDRGN conformational axis)")
    ax.set_ylabel("density")
    ax.set_title("PC1 marginal: crude tertile cuts (red) vs 3-component GMM")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# --------------------------------------------------------------------------- #
# .cs export
# --------------------------------------------------------------------------- #
def save_cs(path, arr):
    with open(path, "wb") as fh:
        np.save(fh, arr)


def export_subsets(prefix, hard, uids, mask, uid_to_row, cs, labels, outdir,
                   summary_rows):
    """Write one .cs per class for the particles selected by `mask`."""
    k = len(labels)
    for j in range(k):
        sel = (hard == j) & mask
        sel_uids = uids[sel]
        rows = [uid_to_row[int(u)] for u in sel_uids.tolist() if int(u) in uid_to_row]
        rows = np.asarray(rows, dtype=np.intp)
        out = os.path.join(outdir, f"{prefix}_{labels[j]}.cs")
        save_cs(out, cs[rows])
        summary_rows.append((prefix, labels[j], len(rows)))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--z", required=True, help="cryoDRGN z.N.pkl latent embedding.")
    p.add_argument("--passthrough-cs", required=True,
                   help="passthrough .cs (uid order of z; blob/pose/CTF for export).")
    p.add_argument("--cs", required=True,
                   help="particles .cs with CryoSPARC class posteriors.")
    p.add_argument("--n-dummies", type=int, default=6)
    p.add_argument("--protein-idx", type=int, nargs="*", default=None)
    p.add_argument("-k", "--k", type=int, default=None,
                   help="number of PC1 classes (default: #protein classes).")
    p.add_argument("--gmm-conf", type=float, default=0.8,
                   help="responsibility threshold for the 'confident' PC1-GMM subset.")
    p.add_argument("--latent-npz", default=None,
                   help="optional per_particle.npz from cryodrgn_latent_gmm.py to "
                        "include the full-latent-GMM partition in the overlap table.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-o", "--outdir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # --- load + align latent to CryoSPARC posteriors by uid -----------------
    z = clg.load_latent(args.z)
    z, cryo_post, cryo_hard, uid, n_protein = clg.align_z_to_posteriors(
        z, args.passthrough_cs, args.cs, args.n_dummies, args.protein_idx)
    k = args.k or n_protein
    protein_idx = args.protein_idx or list(range(6, 6 + n_protein))
    labels = [f"P{protein_idx[i]}" for i in range(k)] if k == n_protein \
        else [f"S{i}" for i in range(k)]
    N = len(z)

    # --- PC1 (dominant axis of standardized latent) -------------------------
    Xs = StandardScaler().fit_transform(z)
    pca = PCA(n_components=min(5, z.shape[1]), random_state=args.seed).fit(Xs)
    scores = pca.transform(Xs)
    pc1 = scores[:, 0]
    print(f"[pc1] PC1 explained variance ratio = {pca.explained_variance_ratio_[0]:.3f}")

    # --- RUN 1: crude division along PC1 (equal-population tertiles) ---------
    edges = np.quantile(pc1, np.linspace(0, 1, k + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    pc1_crude = np.clip(np.digitize(pc1, edges[1:-1]), 0, k - 1).astype(int)
    pc1_crude, _ = align_labels_to_reference(cryo_hard, pc1_crude, k)

    # --- RUN 2: 1-D K-component GMM on PC1 ----------------------------------
    gmm1d = GaussianMixture(k, covariance_type="full", n_init=10, max_iter=1000,
                            tol=1e-6, random_state=args.seed).fit(pc1[:, None])
    resp1d = gmm1d.predict_proba(pc1[:, None])
    pc1_gmm = resp1d.argmax(axis=1)
    pc1_gmm, remap = align_labels_to_reference(cryo_hard, pc1_gmm, k)
    resp1d = resp1d[:, np.argsort(remap)]                 # reorder cols to aligned labels
    conf = resp1d.max(axis=1)
    conf_mask = conf >= args.gmm_conf
    print(f"[pc1-gmm] confident particles (resp>={args.gmm_conf}): "
          f"{int(conf_mask.sum()):,} / {N:,} ({100*conf_mask.mean():.1f}%)")

    # plot PC1 marginal with crude cuts + GMM bells (labels ordered by PC1 mean)
    pc1_hist(pc1, gmm1d, edges, labels, os.path.join(args.outdir, "pc1_marginal.png"))

    # --- partitions to compare ----------------------------------------------
    partitions = {
        "cryosparc": cryo_hard,
        "pc1_crude": pc1_crude,
        "pc1_gmm": pc1_gmm,
    }
    if args.latent_npz and os.path.exists(args.latent_npz):
        d = np.load(args.latent_npz)
        npz_uid = d["uid"].astype(np.uint64)
        lat_hard = d["cryodrgn_gmm_posterior"].argmax(axis=1)
        row_of = {int(u): i for i, u in enumerate(npz_uid.tolist())}
        lat_aligned = np.full(N, -1, dtype=int)
        for i, u in enumerate(uid.tolist()):
            r = row_of.get(int(u))
            if r is not None:
                lat_aligned[i] = lat_hard[r]
        if (lat_aligned >= 0).all():
            partitions["latent_gmm"] = lat_aligned
            print("[overlap] included full latent-GMM partition from", args.latent_npz)
        else:
            print("[overlap] latent_npz did not cover all uids; skipping latent_gmm")

    # --- particle overlap: pairwise ARI/AMI/NMI -----------------------------
    names = list(partitions)
    overlap = {"ari": {}, "ami": {}, "nmi": {}}
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            na, nb = names[a], names[b]
            key = f"{na}__vs__{nb}"
            overlap["ari"][key] = float(adjusted_rand_score(partitions[na], partitions[nb]))
            overlap["ami"][key] = float(adjusted_mutual_info_score(partitions[na], partitions[nb]))
            overlap["nmi"][key] = float(normalized_mutual_info_score(partitions[na], partitions[nb]))

    # --- confusion matrices vs CryoSPARC ------------------------------------
    C_crude = row_normalised_confusion(cryo_hard, pc1_crude, k)
    C_gmm = row_normalised_confusion(cryo_hard, pc1_gmm, k)
    heatmap(C_crude, labels, labels,
            "PC1 crude vs CryoSPARC  P(PC1=j | CryoSPARC=i)",
            os.path.join(args.outdir, "confusion_pc1crude_vs_cryosparc.png"),
            xlabel="PC1-crude class", ylabel="CryoSPARC class")
    heatmap(C_gmm, labels, labels,
            "PC1 GMM vs CryoSPARC  P(PC1=j | CryoSPARC=i)",
            os.path.join(args.outdir, "confusion_pc1gmm_vs_cryosparc.png"),
            xlabel="PC1-GMM class", ylabel="CryoSPARC class")

    # --- cryoDRGN witness metric: self soft-posterior confusion of PC1-GMM ---
    # Analogue of CryoSPARC's hetero-refinement class-population confusion: how
    # much does each PC1-GMM class "leak" into the others under its own soft
    # responsibilities. Diagonal ~1 => clean separation; off-diagonal => overlap.
    C_witness = soft_posterior_confusion(resp1d)
    heatmap(C_witness, labels, labels,
            "cryoDRGN PC1-GMM witness confusion (soft)",
            os.path.join(args.outdir, "confusion_pc1gmm_witness.png"),
            xlabel="assigned-as", ylabel="true class (soft)")

    # --- export .cs subsets for ab-initio -> NU -----------------------------
    cs_pass = np.load(args.passthrough_cs)
    pass_uids = cs_pass["uid"].astype(np.uint64)
    uid_to_row = {int(u): i for i, u in enumerate(pass_uids.tolist())}

    summary_rows = []
    all_mask = np.ones(N, dtype=bool)
    export_subsets("pc1_crude", pc1_crude, uid, all_mask, uid_to_row, cs_pass,
                   labels, args.outdir, summary_rows)       # RUN 1
    export_subsets("pc1_gmm", pc1_gmm, uid, all_mask, uid_to_row, cs_pass,
                   labels, args.outdir, summary_rows)        # RUN 2 (all)
    export_subsets("pc1_gmm_conf", pc1_gmm, uid, conf_mask, uid_to_row, cs_pass,
                   labels, args.outdir, summary_rows)        # RUN 2 (confident)

    # --- per-particle assignment table --------------------------------------
    with open(os.path.join(args.outdir, "assignments.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uid", "pc1", "pc1_crude", "pc1_gmm", "pc1_gmm_conf",
                    "pc1_gmm_maxresp", "cryosparc_hard"])
        for i in range(N):
            w.writerow([int(uid[i]), f"{pc1[i]:.5f}", labels[pc1_crude[i]],
                        labels[pc1_gmm[i]], int(conf_mask[i]),
                        f"{conf[i]:.4f}", labels[cryo_hard[i]]])

    # --- numeric summary -----------------------------------------------------
    crude_counts = {labels[j]: int((pc1_crude == j).sum()) for j in range(k)}
    gmm_counts = {labels[j]: int((pc1_gmm == j).sum()) for j in range(k)}
    gmm_conf_counts = {labels[j]: int(((pc1_gmm == j) & conf_mask).sum()) for j in range(k)}
    cryo_counts = {labels[j]: int((cryo_hard == j).sum()) for j in range(k)}

    result = {
        "n_particles": int(N),
        "k": int(k),
        "labels": labels,
        "pc1_explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
        "gmm_conf_threshold": float(args.gmm_conf),
        "n_confident": int(conf_mask.sum()),
        "frac_confident": float(conf_mask.mean()),
        "counts": {
            "cryosparc": cryo_counts,
            "pc1_crude": crude_counts,
            "pc1_gmm": gmm_counts,
            "pc1_gmm_conf": gmm_conf_counts,
        },
        "overlap": overlap,
        "confusion_pc1crude_vs_cryosparc": C_crude.tolist(),
        "confusion_pc1gmm_vs_cryosparc": C_gmm.tolist(),
        "witness_confusion_pc1gmm": C_witness.tolist(),
        "gmm1d_means": gmm1d.means_.ravel().tolist(),
        "gmm1d_sds": np.sqrt(gmm1d.covariances_.ravel()).tolist(),
        "gmm1d_weights": gmm1d.weights_.ravel().tolist(),
    }
    with open(os.path.join(args.outdir, "overlap_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)

    # --- human summary -------------------------------------------------------
    lines = [
        "# cryoDRGN PC1 classification, overlap, and ab-initio/NU export sets",
        "",
        f"- particles: {N:,}  |  classes (k): {k}  |  labels: {', '.join(labels)}",
        f"- PC1 explained variance ratio: {pca.explained_variance_ratio_[0]:.3f}",
        f"- PC1-GMM confident (resp >= {args.gmm_conf}): {int(conf_mask.sum()):,} "
        f"({100*conf_mask.mean():.1f}%)",
        "",
        "## Class counts",
        "",
        "| class | CryoSPARC | PC1 crude | PC1 GMM (all) | PC1 GMM (confident) |",
        "|---|---|---|---|---|",
    ]
    for j in range(k):
        lab = labels[j]
        lines.append(f"| {lab} | {cryo_counts[lab]:,} | {crude_counts[lab]:,} | "
                     f"{gmm_counts[lab]:,} | {gmm_conf_counts[lab]:,} |")
    lines += [
        "",
        "## Particle overlap (adjusted Rand index; 1=identical, 0=chance)",
        "",
        "| partitions | ARI | AMI | NMI |",
        "|---|---|---|---|",
    ]
    for key in overlap["ari"]:
        lines.append(f"| {key.replace('__vs__', ' vs ')} | "
                     f"{overlap['ari'][key]:.3f} | {overlap['ami'][key]:.3f} | "
                     f"{overlap['nmi'][key]:.3f} |")
    lines += [
        "",
        "## The 3 ab-initio -> NU runs (import each .cs, Ab-initio init 12 A / final 4 A, then NU)",
        "",
        "1. **RUN 1 - crude PC1 division**:  pc1_crude_P*.cs  (equal-population PC1 tertiles)",
        "2. **RUN 2 - PC1 3-comp GMM, confident only**:  pc1_gmm_conf_P*.cs "
        "(also pc1_gmm_P*.cs = all hard-assigned)",
        "3. **RUN 3 - full latent-space GMM**:  cryodrgn_class_P*.cs "
        "(from export_cryodrgn_subsets.py)",
        "",
        "## .cs files written",
        "",
    ]
    for prefix, lab, n in summary_rows:
        lines.append(f"- {prefix}_{lab}.cs : {n:,} particles")
    with open(os.path.join(args.outdir, "SUMMARY.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n[done] outputs ->", args.outdir)
    print("  overlap_metrics.json, assignments.csv, SUMMARY.md, confusion_*.png, pc1_marginal.png")
    for prefix, lab, n in summary_rows:
        print(f"  {prefix}_{lab}.cs : {n:,}")


if __name__ == "__main__":
    main()
