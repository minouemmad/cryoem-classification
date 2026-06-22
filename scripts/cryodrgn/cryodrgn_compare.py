#!/usr/bin/env python
"""Compare a cryoDRGN latent classification against reference labels.

Clusters the latent z (KMeans + GMM) and scores the clusters against any of:
  --gt-labels PKL   ground-truth integer labels (synthetic demo)
  --cs PATH         CryoSPARC hard class (argmax of class_posterior); --protein-only
  --gmm-resp NPY    existing GMM responsibilities (argmax -> label)

Labels align to z by row index (cryoDRGN preserves particle order; pass --ind if a
subset was trained). Writes ARI/AMI/NMI/V-measure, confusion heatmaps, latent plots,
metrics.csv and SUMMARY.md.
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             normalized_mutual_info_score, v_measure_score)
from sklearn.mixture import GaussianMixture


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workdir", required=True, help="cryoDRGN train_vae output dir.")
    p.add_argument("--epoch", type=int, help="Epoch z to load (default: latest).")
    p.add_argument("--z", help="Explicit z .pkl path (overrides --workdir/--epoch).")
    p.add_argument("-k", "--k", type=int, help="Number of clusters (default: #ref classes).")
    p.add_argument("--ind", help="PKL of training-subset indices.")
    p.add_argument("--gt-labels", help="PKL of ground-truth labels.")
    p.add_argument("--cs", help="CryoSPARC particles .cs for reference classes.")
    p.add_argument("--n-dummies", type=int, default=6)
    p.add_argument("--protein-only", action="store_true")
    p.add_argument("--gmm-resp", help="NPY responsibility matrix.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-o", "--outdir", required=True)
    return p.parse_args()


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_z(args):
    path = args.z
    if path is None and args.epoch is not None:
        path = os.path.join(args.workdir, f"z.{args.epoch}.pkl")
    if path is None:
        zs = [f for f in os.listdir(args.workdir)
              if f.startswith("z.") and f.endswith(".pkl") and f != "z.pkl"]
        path = os.path.join(args.workdir, max(zs, key=lambda f: int(f.split(".")[1]))
                            if zs else "z.pkl")
    z = np.asarray(load_pkl(path), dtype=np.float64)
    print(f"[compare] loaded z {z.shape} from {path}")
    return z


def score(ref, pred):
    return {"ARI": adjusted_rand_score(ref, pred),
            "AMI": adjusted_mutual_info_score(ref, pred),
            "NMI": normalized_mutual_info_score(ref, pred),
            "Vmeasure": v_measure_score(ref, pred)}


def hungarian_confusion(ref, pred):
    rl, pl = np.unique(ref), np.unique(pred)
    R = {c: i for i, c in enumerate(rl)}
    P = {c: i for i, c in enumerate(pl)}
    M = np.zeros((len(rl), len(pl)), int)
    for r, q in zip(ref, pred):
        M[R[r], P[q]] += 1
    _, cols = linear_sum_assignment(-M)
    order = list(cols) + [c for c in range(M.shape[1]) if c not in cols]
    purity = M.max(axis=0).sum() / M.sum() if M.sum() else 0.0
    return M[:, order], rl, [pl[c] for c in order], purity


def embed_2d(z, seed):
    if z.shape[1] == 1:
        return np.column_stack([z[:, 0], np.zeros(len(z))]), "z"
    if z.shape[1] == 2:
        return z, "z"
    import umap
    return umap.UMAP(n_components=2, random_state=seed).fit_transform(z), "UMAP"


def scatter(emb, labels, title, path, axlabel):
    plt.figure(figsize=(6, 5))
    labels = np.asarray(labels)
    for c in np.unique(labels):
        m = labels == c
        plt.scatter(emb[m, 0], emb[m, 1], s=4, alpha=0.5, label=str(c))
    plt.title(title); plt.xlabel(f"{axlabel} 1"); plt.ylabel(f"{axlabel} 2")
    plt.legend(markerscale=3, fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def heatmap(M, rl, cl, title, path):
    plt.figure(figsize=(1.2 + 0.8 * M.shape[1], 1.2 + 0.8 * M.shape[0]))
    plt.imshow(M, cmap="Blues", aspect="auto"); plt.colorbar(label="count")
    plt.xticks(range(M.shape[1]), [f"c{c}" for c in cl])
    plt.yticks(range(M.shape[0]), [str(r) for r in rl])
    plt.xlabel("cryoDRGN cluster"); plt.ylabel("reference")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            plt.text(j, i, int(M[i, j]), ha="center", va="center",
                     color="white" if M[i, j] > M.max() / 2 else "black", fontsize=8)
    plt.title(title); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def build_references(args):
    refs, protein_idx = {}, None
    if args.gt_labels:
        refs["ground_truth"] = np.asarray(load_pkl(args.gt_labels)).astype(int)
    if args.cs:
        from gmm_pipeline.data_io import load_posteriors
        post = load_posteriors(args.cs, n_dummies=args.n_dummies)
        refs["cryosparc_class"] = post.hard_class.astype(int)
        protein_idx = post.protein_idx
    if args.gmm_resp:
        refs["existing_gmm"] = np.load(args.gmm_resp).argmax(axis=1).astype(int)
    return refs, protein_idx


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    z = load_z(args)
    n_z = len(z)
    ind = np.asarray(load_pkl(args.ind)).astype(int) if args.ind else None

    refs, protein_idx = build_references(args)
    aligned = {}
    for name, lab in refs.items():
        if ind is not None and len(lab) != n_z and len(lab) >= ind.max() + 1:
            lab = lab[ind]
        if len(lab) != n_z:
            print(f"[compare] WARNING: '{name}' length {len(lab)} != z {n_z}; skipping.")
            continue
        aligned[name] = lab
    if not aligned:
        sys.exit("[compare] No usable references (--gt-labels / --cs / --gmm-resp).")

    k = args.k or max(2, len(np.unique(next(iter(aligned.values())))))
    print(f"[compare] clustering z into k={k} (KMeans + GMM)")
    clusterings = {
        "kmeans": KMeans(n_clusters=k, n_init=10, random_state=args.seed).fit_predict(z),
        "gmm": GaussianMixture(k, covariance_type="full",
                               random_state=args.seed).fit(z).predict(z),
    }

    emb, axlabel = embed_2d(z, args.seed)
    for name, lab in {**clusterings, **{f"ref_{r}": v for r, v in aligned.items()}}.items():
        scatter(emb, lab, f"latent z — {name}",
                os.path.join(args.outdir, f"latent_{name}.png"), axlabel)

    rows = []
    summary = ["# cryoDRGN classification comparison", "",
               f"- latent z: {n_z} particles, zdim={z.shape[1]}",
               f"- clusters: k={k} (KMeans + GMM)", f"- plot embedding: {axlabel}", "",
               "## Agreement (1.0 = identical, 0.0 = chance)", "",
               "| reference | clustering | ARI | AMI | NMI | V-measure |",
               "|---|---|---|---|---|---|"]
    for rname, ref in aligned.items():
        for cname, pred in clusterings.items():
            r_use, p_use = ref, pred
            if rname == "cryosparc_class" and args.protein_only and protein_idx is not None:
                mask = np.isin(ref, protein_idx)
                r_use, p_use = ref[mask], pred[mask]
            s = score(r_use, p_use)
            rows.append({"reference": rname, "clustering": cname, **s})
            summary.append(f"| {rname} | {cname} | {s['ARI']:.3f} | {s['AMI']:.3f} | "
                           f"{s['NMI']:.3f} | {s['Vmeasure']:.3f} |")
            M, rl, cl, purity = hungarian_confusion(r_use, p_use)
            heatmap(M, rl, cl, f"{rname} vs {cname} (purity={purity:.2f})",
                    os.path.join(args.outdir, f"confusion_{rname}_{cname}.png"))

    with open(os.path.join(args.outdir, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["reference", "clustering",
                                          "ARI", "AMI", "NMI", "Vmeasure"])
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(args.outdir, "SUMMARY.md"), "w") as f:
        f.write("\n".join(summary) + "\n")

    print(f"[compare] wrote results to {args.outdir}")
    for r in rows:
        print(f"  {r['reference']:>16s} vs {r['clustering']:<7s}  ARI={r['ARI']:.3f} "
              f"AMI={r['AMI']:.3f} NMI={r['NMI']:.3f}")


if __name__ == "__main__":
    main()
