#!/usr/bin/env python
"""Compare a cryoDRGN latent-space classification against reference labels.

Given a trained cryoDRGN workdir, this clusters the per-particle latent embeddings
(z) with KMeans and a Gaussian mixture, then scores those clusters against one or
more reference labellings:

  * --gt-labels PKL    ground-truth integer labels (used by the synthetic demo)
  * --cs PATH          CryoSPARC per-particle hard class (argmax of
                       alignments3D_multi/class_posterior); optionally protein-only
  * --gmm-resp NPY     an existing GMM responsibility matrix (argmax -> label)

cryoDRGN preserves the input particle order, so reference labels are aligned to z by
ROW INDEX. If the model was trained on a subset, pass the same --ind PKL so the
references are subset identically.

Reported metrics (cluster vs each reference): Adjusted Rand Index, Adjusted Mutual
Information, Normalized Mutual Information, V-measure, plus a Hungarian-aligned
confusion matrix. Writes plots (UMAP/PCA of z coloured by each labelling, confusion
heatmaps), a metrics CSV, and SUMMARY.md.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np

# Allow importing the repo's gmm_pipeline package when run from anywhere.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score,
)
from sklearn.mixture import GaussianMixture


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workdir", required=True, help="cryoDRGN train_vae output dir.")
    p.add_argument("--epoch", type=int, default=None,
                   help="Epoch to load z from (default: latest z.*.pkl, else z.pkl).")
    p.add_argument("--z", default=None,
                   help="Explicit path to a z .pkl (overrides --workdir/--epoch).")
    p.add_argument("-k", "--k", type=int, default=None,
                   help="Number of latent clusters (default: #classes in first "
                        "reference, else 3).")
    p.add_argument("--ind", default=None,
                   help="PKL of training-subset indices, to subset references.")
    p.add_argument("--gt-labels", default=None,
                   help="PKL of ground-truth integer labels (demo).")
    p.add_argument("--cs", default=None,
                   help="CryoSPARC particles .cs for reference hard classes.")
    p.add_argument("--n-dummies", type=int, default=6,
                   help="Leading dummy classes in --cs (default 6).")
    p.add_argument("--protein-only", action="store_true",
                   help="Restrict the --cs comparison to protein-class particles "
                        "(masks z and labels identically to keep row alignment).")
    p.add_argument("--gmm-resp", default=None,
                   help="NPY responsibility matrix (N,K); argmax -> reference label.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-o", "--outdir", required=True)
    return p.parse_args()


def load_z(args) -> np.ndarray:
    path = args.z
    if path is None:
        if args.epoch is not None:
            path = os.path.join(args.workdir, f"z.{args.epoch}.pkl")
        else:
            cands = [f for f in os.listdir(args.workdir)
                     if f.startswith("z.") and f.endswith(".pkl")]
            if cands:
                def ep(f):
                    try:
                        return int(f.split(".")[1])
                    except ValueError:
                        return -1
                path = os.path.join(args.workdir, sorted(cands, key=ep)[-1])
            else:
                path = os.path.join(args.workdir, "z.pkl")
    with open(path, "rb") as f:
        z = pickle.load(f)
    z = np.asarray(z, dtype=np.float64)
    print(f"[compare] loaded z {z.shape} from {path}")
    return z


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def hungarian_confusion(ref: np.ndarray, pred: np.ndarray):
    """Confusion matrix with predicted clusters permuted to best match references."""
    ref_labels = np.unique(ref)
    pred_labels = np.unique(pred)
    R = {c: i for i, c in enumerate(ref_labels)}
    P = {c: i for i, c in enumerate(pred_labels)}
    M = np.zeros((len(ref_labels), len(pred_labels)), dtype=int)
    for r, p in zip(ref, pred):
        M[R[r], P[p]] += 1
    # maximize matched count -> minimize negative
    rows, cols = linear_sum_assignment(-M)
    col_order = list(cols) + [c for c in range(M.shape[1]) if c not in cols]
    M_aligned = M[:, col_order]
    purity = M.max(axis=0).sum() / M.sum() if M.sum() else 0.0
    return M_aligned, ref_labels, [pred_labels[c] for c in col_order], purity


def score(ref: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "ARI": adjusted_rand_score(ref, pred),
        "AMI": adjusted_mutual_info_score(ref, pred),
        "NMI": normalized_mutual_info_score(ref, pred),
        "Vmeasure": v_measure_score(ref, pred),
    }


def embed_2d(z: np.ndarray, seed: int) -> tuple[np.ndarray, str]:
    if z.shape[1] == 1:
        return np.column_stack([z[:, 0], np.zeros(len(z))]), "z (1D)"
    if z.shape[1] == 2:
        return z, "z"
    try:
        import umap
        emb = umap.UMAP(n_components=2, random_state=seed).fit_transform(z)
        return emb, "UMAP"
    except Exception as e:  # pragma: no cover
        print(f"[compare] UMAP unavailable ({e}); using PCA.")
        from sklearn.decomposition import PCA
        return PCA(n_components=2).fit_transform(z), "PCA"


def scatter(emb, labels, title, path, xlabel):
    plt.figure(figsize=(6, 5))
    labels = np.asarray(labels)
    for c in np.unique(labels):
        m = labels == c
        plt.scatter(emb[m, 0], emb[m, 1], s=4, alpha=0.5, label=str(c))
    plt.title(title)
    plt.xlabel(f"{xlabel} 1")
    plt.ylabel(f"{xlabel} 2")
    plt.legend(markerscale=3, fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def heatmap(M, row_labels, col_labels, title, path):
    plt.figure(figsize=(1.2 + 0.8 * M.shape[1], 1.2 + 0.8 * M.shape[0]))
    plt.imshow(M, cmap="Blues", aspect="auto")
    plt.colorbar(label="count")
    plt.xticks(range(M.shape[1]), [f"c{c}" for c in col_labels])
    plt.yticks(range(M.shape[0]), [str(r) for r in row_labels])
    plt.xlabel("cryoDRGN cluster (aligned)")
    plt.ylabel("reference label")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            plt.text(j, i, int(M[i, j]), ha="center", va="center",
                     color="black" if M[i, j] < M.max() / 2 else "white", fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def build_references(args, n_z: int):
    """Return dict name -> full-length (n_z,) label array (pre-subset)."""
    refs = {}
    if args.gt_labels:
        refs["ground_truth"] = np.asarray(load_pkl(args.gt_labels)).astype(int)
    if args.cs:
        from gmm_pipeline.data_io import load_posteriors
        post = load_posteriors(args.cs, n_dummies=args.n_dummies)
        refs["cryosparc_class"] = post.hard_class.astype(int)
        refs["_protein_idx"] = post.protein_idx
    if args.gmm_resp:
        resp = np.load(args.gmm_resp)
        refs["existing_gmm"] = np.asarray(resp).argmax(axis=1).astype(int)
    return refs


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng_seed = args.seed

    z = load_z(args)
    n_z = len(z)

    ind = None
    if args.ind:
        ind = np.asarray(load_pkl(args.ind)).astype(int)

    refs = build_references(args, n_z)
    protein_idx = refs.pop("_protein_idx", None)

    # Subset references to the training indices if needed, and validate lengths.
    aligned = {}
    for name, lab in refs.items():
        if ind is not None and len(lab) != n_z and len(lab) >= ind.max() + 1:
            lab = lab[ind]
        if len(lab) != n_z:
            print(f"[compare] WARNING: reference '{name}' length {len(lab)} != "
                  f"z length {n_z}; skipping. (Pass --ind if a subset was trained.)")
            continue
        aligned[name] = lab

    if not aligned:
        print("[compare] No usable references. Provide --gt-labels / --cs / --gmm-resp.")
        return 1

    # number of clusters
    if args.k is not None:
        k = args.k
    else:
        first = next(iter(aligned.values()))
        k = max(2, len(np.unique(first)))
    print(f"[compare] clustering z into k={k} clusters (KMeans + GMM)")

    km = KMeans(n_clusters=k, n_init=10, random_state=rng_seed).fit_predict(z)
    gm = GaussianMixture(n_components=k, covariance_type="full",
                         random_state=rng_seed).fit(z).predict(z)
    clusterings = {"kmeans": km, "gmm": gm}

    emb, emb_name = embed_2d(z, rng_seed)

    # plots of the latent space coloured by each clustering + reference
    for cname, lab in clusterings.items():
        scatter(emb, lab, f"latent z — {cname} (k={k})",
                os.path.join(args.outdir, f"latent_{cname}.png"), emb_name)
    for rname, lab in aligned.items():
        scatter(emb, lab, f"latent z — {rname}",
                os.path.join(args.outdir, f"latent_ref_{rname}.png"), emb_name)

    # metrics
    rows = []
    summary = ["# cryoDRGN classification comparison", ""]
    summary.append(f"- latent z: {z.shape[0]} particles, zdim={z.shape[1]}")
    summary.append(f"- clusters: k={k} (KMeans + Gaussian mixture)")
    summary.append(f"- embedding for plots: {emb_name}")
    summary.append("")
    summary.append("## Agreement metrics (1.0 = identical partition, 0.0 = chance)")
    summary.append("")
    summary.append("| reference | clustering | ARI | AMI | NMI | V-measure |")
    summary.append("|---|---|---|---|---|---|")

    for rname, ref in aligned.items():
        ref_use = ref
        for cname, pred in clusterings.items():
            pred_use = pred
            if rname == "cryosparc_class" and args.protein_only and protein_idx is not None:
                mask = np.isin(ref, protein_idx)
                ref_use = ref[mask]
                pred_use = pred[mask]
            s = score(ref_use, pred_use)
            rows.append({"reference": rname, "clustering": cname, **s})
            summary.append(
                f"| {rname} | {cname} | {s['ARI']:.3f} | {s['AMI']:.3f} | "
                f"{s['NMI']:.3f} | {s['Vmeasure']:.3f} |")
            M, rl, cl, purity = hungarian_confusion(ref_use, pred_use)
            heatmap(M, rl, cl, f"{rname} vs {cname} (purity={purity:.2f})",
                    os.path.join(args.outdir, f"confusion_{rname}_{cname}.png"))

    # CSV
    import csv
    with open(os.path.join(args.outdir, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["reference", "clustering",
                                          "ARI", "AMI", "NMI", "Vmeasure"])
        w.writeheader()
        w.writerows(rows)

    summary.append("")
    summary.append("## Files")
    summary.append("- metrics.csv — all scores")
    summary.append("- latent_*.png — z embedding coloured by each clustering/reference")
    summary.append("- confusion_*.png — Hungarian-aligned confusion matrices")
    with open(os.path.join(args.outdir, "SUMMARY.md"), "w") as f:
        f.write("\n".join(summary) + "\n")

    print(f"\n[compare] wrote results to {args.outdir}")
    for r in rows:
        print(f"  {r['reference']:>16s} vs {r['clustering']:<7s}  "
              f"ARI={r['ARI']:.3f} AMI={r['AMI']:.3f} NMI={r['NMI']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
