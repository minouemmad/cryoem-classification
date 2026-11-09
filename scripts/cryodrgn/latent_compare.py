#!/usr/bin/env python
"""Compare two cryoDRGN latents (e.g. VAE vs RAE on the same particles) as spaces to cluster in:
class signal (kNN purity, held-out accuracy), Fisher ratio, shape vs a Gaussian null, how uniformly z distance tracks
decoded-volume distance, and canonical correlations between the two latents. Also writes 2-D views.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mrcfile
import numpy as np
from scipy.stats import spearmanr
from sklearn.cross_decomposition import CCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from cryo_utils import load_class_set, load_latents


def twonn(X):
    d, _ = NearestNeighbors(n_neighbors=3).fit(X).kneighbors(X)
    mu = d[:, 2] / np.clip(d[:, 1], 1e-12, None)
    mu = np.sort(mu[np.isfinite(mu) & (mu > 1)])
    F = np.arange(1, len(mu) + 1) / len(mu)
    keep = F < 0.9
    x, y = np.log(mu[keep]), -np.log(1 - F[keep])
    return float((x * y).sum() / (x * x).sum())


def logdens_spread(X, k=30):
    d = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)[0][:, -1]
    ld = -X.shape[1] * np.log(d)
    return float(np.percentile(ld, 95) - np.percentile(ld, 5))


def shape_stats(X, rng):
    ev = np.linalg.eigvalsh(np.cov(X.T))
    return dict(participation_ratio=float(ev.sum() ** 2 / (ev ** 2).sum()), twonn=twonn(X),
                logdens_p95_p5=logdens_spread(X))


def purity(X, y, ks):
    nn = NearestNeighbors(n_neighbors=max(ks) + 1).fit(X)
    idx = nn.kneighbors(X, return_distance=False)[:, 1:]
    same = (y[idx] == y[:, None])
    p = np.bincount(y) / len(y)
    chance = float((p ** 2).sum())
    return {int(k): float(same[:, :k].mean() / chance) for k in ks}, chance


def load_vols(dec: Path, box_ds=32):
    files = sorted((dec / "volumes").glob("vol_*.mrc"), key=lambda f: int(f.stem.split("_")[1]))
    V = []
    for f in files:
        v = mrcfile.read(f).astype(np.float32)
        b = v.shape[0] // box_ds
        v = v.reshape(box_ds, b, box_ds, b, box_ds, b).mean((1, 3, 5))
        V.append(v)
    V = np.stack(V)
    g = np.linspace(-1, 1, box_ds)
    r = np.sqrt(sum(a ** 2 for a in np.meshgrid(g, g, g, indexing="ij")))
    return V[:, r < 0.85]


def z_to_volume(C, V, k=10):
    from scipy.spatial.distance import pdist
    dz, dv = pdist(C), pdist(V)
    rho = float(spearmanr(dz, dv)[0])
    nz = NearestNeighbors(n_neighbors=k + 1).fit(C).kneighbors(C, return_distance=False)[:, 1:]
    nv = NearestNeighbors(n_neighbors=k + 1).fit(V).kneighbors(V, return_distance=False)[:, 1:]
    trust = float(np.mean([len(set(a) & set(b)) / k for a, b in zip(nz, nv)]))
    i = np.repeat(np.arange(len(C)), 5)
    j = NearestNeighbors(n_neighbors=6).fit(C).kneighbors(C, return_distance=False)[:, 1:].ravel()
    ratio = np.linalg.norm(V[i] - V[j], axis=1) / np.linalg.norm(C[i] - C[j], axis=1)
    return dict(spearman_dz_dv=rho, knn10_overlap=trust,
                cv_dv_over_dz=float(ratio.std() / ratio.mean()),
                dv_over_dz_p10_p90=[float(np.percentile(ratio, 10)), float(np.percentile(ratio, 90))])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, nargs=2, required=True)
    p.add_argument("--labels", nargs=2, default=["VAE", "RAE"])
    p.add_argument("--decoded", type=Path, nargs=2, required=True)
    p.add_argument("--epoch", type=int, default=50)
    p.add_argument("--particles-cs", type=Path, required=True)
    p.add_argument("--class-set", nargs="+", required=True)
    p.add_argument("--n", type=int, default=40000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-o", "--out", type=Path, required=True)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    cs = np.load(a.particles_cs)
    _, names, cls = load_class_set(a.class_set, cs["uid"])
    Z = [load_latents(r / f"z.{a.epoch}.pkl") for r in a.runs]
    idx = rng.choice(len(cls), a.n, replace=False)
    idx = idx[cls[idx] >= 0]
    y = cls[idx]
    res = dict(classes=names, labels=a.labels, n=int(len(idx)), models={})
    views = dict(idx=idx, y=y)
    import umap
    for lab, z, dec in zip(a.labels, Z, a.decoded):
        X = z[idx]
        G = rng.multivariate_normal(z.mean(0), np.cov(z.T), len(X))
        r = dict()
        r["purity"], r["purity_chance"] = purity(X, y, (5, 15, 50, 150))
        r["knn15_bal_acc"] = float(balanced_accuracy_score(
            y, cross_val_predict(KNeighborsClassifier(15), X, y, cv=5)))
        r["linear_bal_acc"] = float(balanced_accuracy_score(
            y, cross_val_predict(LogisticRegression(max_iter=2000), X, y, cv=5)))
        mu = X.mean(0)
        Sb = sum((y == c).sum() * np.outer(X[y == c].mean(0) - mu, X[y == c].mean(0) - mu) for c in np.unique(y))
        Sw = sum(np.cov(X[y == c].T) * ((y == c).sum() - 1) for c in np.unique(y))
        r["fisher_ratio"] = float(np.trace(Sb) / np.trace(Sw))
        r["shape"] = shape_stats(X, rng)
        r["shape_null"] = shape_stats(G, rng)
        C = np.loadtxt(dec / "centers.txt")
        V = load_vols(dec)
        r["z_to_volume"] = z_to_volume(C, V)
        res["models"][lab] = r
        print(lab, json.dumps(r), flush=True)
        sub = rng.choice(len(X), 20000, replace=False)
        views[f"{lab}_umap"] = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(X[sub])
        views[f"{lab}_umap_rows"] = sub
        views[f"{lab}_lda"] = LinearDiscriminantAnalysis(n_components=2).fit(X, y).transform(X)
    cca = CCA(n_components=Z[0].shape[1], max_iter=2000).fit(Z[0][idx], Z[1][idx])
    A, B = cca.transform(Z[0][idx], Z[1][idx])
    res["cca"] = [float(np.corrcoef(A[:, i], B[:, i])[0, 1]) for i in range(A.shape[1])]
    print("cca", res["cca"])
    a.out.write_text(json.dumps(res, indent=1))
    np.savez(a.out.with_suffix(".npz"), **views)


if __name__ == "__main__":
    main()
