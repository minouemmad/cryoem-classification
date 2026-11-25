#!/usr/bin/env python
"""Cluster one cryoDRGN latent with several algorithms (k-means, GMM, Ward, spectral, Leiden, HDBSCAN) and
K-selection criteria, each calibrated against a Gaussian null with the latent's mean and covariance: a K is only
evidence of clusters if it beats the null. Raw z or its UMAP. Heteroref AMI is reported for evaluation only.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             calinski_harabasz_score, davies_bouldin_score, silhouette_score)
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph

from cryo_utils import load_class_set, load_latents


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def knn_graph(X, k):
    A = kneighbors_graph(X, k, mode="connectivity", include_self=False)
    return ((A + A.T) > 0).astype(float).tocsr()


def spectral_embed(X, k_nn, n_eig):
    """Rows of the first n_eig eigenvectors of the symmetric normalized Laplacian + eigenvalues."""
    A = knn_graph(X, k_nn)
    L = csgraph.laplacian(A, normed=True)
    vals, vecs = eigsh(L, k=n_eig + 1, sigma=-1e-3, which="LM")
    o = np.argsort(vals)
    return vals[o], vecs[:, o]


def transfer(Xfit, lab, Xprobe):
    return KNeighborsClassifier(5).fit(Xfit, lab).predict(Xprobe)


def mean_pair_ari(preds):
    return float(np.mean([adjusted_rand_score(preds[i], preds[j])
                          for i in range(len(preds)) for j in range(i + 1, len(preds))]))


def sep_scores(X, lab, sil_n, rng):
    if len(np.unique(lab)) < 2:
        return dict(silhouette=np.nan, ch=np.nan, db=np.nan)
    s = rng.choice(len(X), min(sil_n, len(X)), replace=False)
    return dict(silhouette=float(silhouette_score(X[s], lab[s])),
                ch=float(calinski_harabasz_score(X, lab)), db=float(davies_bouldin_score(X, lab)))


def kmeans_W(X, lab, centers):
    return float(((X - centers[lab]) ** 2).sum())


def prediction_strength(X, k, rng, splits=5):
    ps = []
    for s in range(splits):
        perm = rng.permutation(len(X))
        A, B = X[perm[: len(X) // 2]], X[perm[len(X) // 2:]]
        ma = KMeans(k, n_init=5, random_state=s).fit(A)
        lb = KMeans(k, n_init=5, random_state=s + 100).fit_predict(B)
        la = ma.predict(B)
        worst = 1.0
        for c in range(k):
            m = lb == c
            n = m.sum()
            if n < 2:
                continue
            counts = np.bincount(la[m], minlength=k)
            worst = min(worst, float((counts * (counts - 1)).sum() / (n * (n - 1))))
        ps.append(worst)
    return float(np.mean(ps))


def icl(gm, X):
    p = gm.predict_proba(X)
    ent = -np.sum(p * np.log(np.clip(p, 1e-300, None)))
    return float(gm.bic(X) + 2 * ent)


def ami(cls, lab):
    keep = cls >= 0
    return float(adjusted_mutual_info_score(cls[keep], lab[keep]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--epoch", type=int, default=50)
    p.add_argument("--rep", choices=["z", "umap"], default="z")
    p.add_argument("--particles-cs", type=Path)
    p.add_argument("--class-set", nargs="+")
    p.add_argument("--labels-npy", type=Path, help="ground-truth labels instead of a class set")
    p.add_argument("--n", type=int, default=20000, help="fit sample (k-means, gmm, leiden, hdbscan)")
    p.add_argument("--n-small", type=int, default=10000, help="fit sample (ward, spectral)")
    p.add_argument("--probe", type=int, default=5000)
    p.add_argument("--kmax", type=int, default=10)
    p.add_argument("--boots", type=int, default=8)
    p.add_argument("--nulls", type=int, default=5)
    p.add_argument("--gap-refs", type=int, default=10)
    p.add_argument("--knn", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-o", "--out", type=Path, required=True)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    z = load_latents(a.run / f"z.{a.epoch}.pkl")
    if a.labels_npy:
        cls_all = np.load(a.labels_npy).astype(int)
        names = [str(c) for c in np.unique(cls_all)]
    else:
        cs = np.load(a.particles_cs)
        _, names, cls_all = load_class_set(a.class_set, cs["uid"])
    assert len(z) == len(cls_all)
    N = len(z)
    mu, cov = z.mean(0), np.cov(z.T)
    Ks = list(range(2, a.kmax + 1))

    # ---- representation: one embedding for the real latent, one per null replicate ----
    pool = rng.choice(N, a.n * (a.boots + 1) + a.probe, replace=False)
    probe_idx = pool[: a.probe]
    samples = [pool[a.probe + b * a.n: a.probe + (b + 1) * a.n] for b in range(a.boots + 1)]
    emb_rows = np.concatenate([probe_idx] + samples)
    if a.rep == "z":
        def embed_real(rows):
            return z[rows]

        def make_null(n, s):
            return np.random.default_rng(1000 + s).multivariate_normal(mu, cov, n)
        Zrep = {int(r): i for i, r in enumerate(emb_rows)}
        E = z[emb_rows]
    else:
        import umap
        fitter = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=a.seed)
        log("umap fit (real)")
        fitter.fit(z[samples[0]])
        E = fitter.transform(z[emb_rows])
        Zrep = {int(r): i for i, r in enumerate(emb_rows)}

        def make_null(n, s):
            G = np.random.default_rng(1000 + s).multivariate_normal(mu, cov, n)
            return umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=s).fit_transform(G)

    def rows_of(idx):
        return E[[Zrep[int(r)] for r in idx]]

    P = rows_of(probe_idx)
    Xs = [rows_of(s) for s in samples]
    X = Xs[0]
    cls = cls_all[samples[0]]
    log("building nulls")
    nulls = [make_null(a.n, s) for s in range(a.nulls)]
    res = dict(params={k: str(v) for k, v in vars(a).items()}, classes=names, n=N,
               rep=a.rep, dim=int(E.shape[1]), methods={})

    def save():
        a.out.write_text(json.dumps(res, indent=1))

    # ---------------- k-means ----------------
    log("kmeans")
    rows = []
    for k in Ks:
        m = KMeans(k, n_init=10, random_state=a.seed).fit(X)
        lab = m.labels_
        r = dict(k=k, **sep_scores(X, lab, 10000, rng))
        preds = [m.predict(P)] + [KMeans(k, n_init=10, random_state=b + 1).fit(Xs[b]).predict(P)
                                  for b in range(1, a.boots + 1)]
        r["stability"] = mean_pair_ari(preds)
        ns = [sep_scores(G, KMeans(k, n_init=10, random_state=a.seed).fit_predict(G), 10000, rng)
              for G in nulls]
        for key in ("silhouette", "ch", "db"):
            r[f"null_{key}"] = float(np.mean([s[key] for s in ns]))
            r[f"null_{key}_sd"] = float(np.std([s[key] for s in ns]))
        logW = np.log(kmeans_W(X, lab, m.cluster_centers_))
        refs = []
        for s in range(a.gap_refs):
            G = make_null(a.n, 500 + s) if a.rep == "z" else nulls[s % a.nulls]
            mg = KMeans(k, n_init=5, random_state=s).fit(G)
            refs.append(np.log(kmeans_W(G, mg.labels_, mg.cluster_centers_)))
        r["gap"] = float(np.mean(refs) - logW)
        r["gap_se"] = float(np.std(refs) * np.sqrt(1 + 1 / len(refs)))
        r["pred_strength"] = prediction_strength(X, k, rng)
        r["null_pred_strength"] = prediction_strength(nulls[0], k, rng)
        r["ami_classes"] = ami(cls, lab)
        rows.append(r)
        log("kmeans", {kk: round(v, 3) for kk, v in r.items()})
    # gap at K = 1 for the 1-SE rule
    logW1 = np.log(((X - X.mean(0)) ** 2).sum())
    refs1 = [np.log(((G - G.mean(0)) ** 2).sum()) for G in
             ([make_null(a.n, 500 + s) for s in range(a.gap_refs)] if a.rep == "z" else nulls)]
    gap1 = dict(k=1, gap=float(np.mean(refs1) - logW1), gap_se=float(np.std(refs1) * np.sqrt(1 + 1 / len(refs1))))
    res["methods"]["kmeans"] = dict(by_k=rows, gap_k1=gap1)
    save()

    # ---------------- GMM ----------------
    log("gmm")
    rows = []
    for k in [1] + Ks:
        gm = GaussianMixture(k, covariance_type="full", n_init=2, random_state=a.seed).fit(X)
        lab = gm.predict(X)
        r = dict(k=k, bic=float(gm.bic(X)), icl=icl(gm, X), mean_maxpost=float(gm.predict_proba(X).max(1).mean()))
        gn = GaussianMixture(k, covariance_type="full", n_init=2, random_state=a.seed).fit(nulls[0])
        r["null_bic"], r["null_icl"] = float(gn.bic(nulls[0])), icl(gn, nulls[0])
        r["null_mean_maxpost"] = float(gn.predict_proba(nulls[0]).max(1).mean())
        if k >= 2:
            r.update(sep_scores(X, lab, 10000, rng))
            preds = [gm.predict(P)] + [GaussianMixture(k, covariance_type="full", random_state=b + 1)
                                       .fit(Xs[b]).predict(P) for b in range(1, min(a.boots, 4) + 1)]
            r["stability"] = mean_pair_ari(preds)
            ns = sep_scores(nulls[0], gn.predict(nulls[0]), 10000, rng)
            r["null_silhouette"] = ns["silhouette"]
            r["ami_classes"] = ami(cls, lab)
        rows.append(r)
        log("gmm", {kk: round(v, 3) for kk, v in r.items()})
    res["methods"]["gmm"] = dict(by_k=rows)
    save()

    # ---------------- Ward ----------------
    log("ward")
    ns_ = a.n_small
    Xw = [Xs[b][:ns_] for b in range(a.boots + 1)]
    links = [linkage(x, "ward") for x in Xw]
    nlinks = [linkage(G[:ns_], "ward") for G in nulls]
    rows = []
    for k in Ks:
        labs = [fcluster(L, k, "maxclust") - 1 for L in links]
        r = dict(k=k, **sep_scores(Xw[0], labs[0], 10000, rng))
        r["stability"] = mean_pair_ari([transfer(Xw[b], labs[b], P) for b in range(a.boots + 1)])
        ns = [sep_scores(G[:ns_], fcluster(L, k, "maxclust") - 1, 10000, rng) for G, L in zip(nulls, nlinks)]
        for key in ("silhouette", "ch", "db"):
            r[f"null_{key}"] = float(np.mean([s[key] for s in ns]))
        r["ami_classes"] = ami(cls[:ns_], labs[0])
        rows.append(r)
        log("ward", {kk: round(v, 3) for kk, v in r.items()})
    res["methods"]["ward"] = dict(by_k=rows)
    save()

    # ---------------- spectral ----------------
    log("spectral")
    n_eig = a.kmax + 2
    embs = [spectral_embed(Xs[b][:ns_], a.knn, n_eig) for b in range(min(a.boots, 4) + 1)]
    nembs = [spectral_embed(G[:ns_], a.knn, n_eig) for G in nulls[:3]]
    vals = embs[0][0]
    rows = []
    for k in Ks:
        labs = []
        for vals_b, vecs_b in embs:
            U = vecs_b[:, :k]
            U = U / np.linalg.norm(U, axis=1, keepdims=True).clip(1e-12)
            labs.append(KMeans(k, n_init=10, random_state=a.seed).fit_predict(U))
        r = dict(k=k, eigengap=float(vals[k] - vals[k - 1]), **sep_scores(Xw[0], labs[0], 10000, rng))
        r["stability"] = mean_pair_ari([transfer(Xs[b][:ns_], labs[b], P) for b in range(len(embs))])
        ng, nsil = [], []
        for (nv, nvec), G in zip(nembs, nulls):
            ng.append(nv[k] - nv[k - 1])
            U = nvec[:, :k] / np.linalg.norm(nvec[:, :k], axis=1, keepdims=True).clip(1e-12)
            nsil.append(sep_scores(G[:ns_], KMeans(k, n_init=10, random_state=0).fit_predict(U), 10000, rng)["silhouette"])
        r["null_eigengap"], r["null_silhouette"] = float(np.mean(ng)), float(np.mean(nsil))
        r["ami_classes"] = ami(cls[:ns_], labs[0])
        rows.append(r)
        log("spectral", {kk: round(v, 4) for kk, v in r.items()})
    res["methods"]["spectral"] = dict(by_k=rows, eigenvalues=vals.tolist(),
                                      null_eigenvalues=[nv.tolist() for nv, _ in nembs])
    save()

    # ---------------- Leiden ----------------
    log("leiden")
    import igraph as ig

    def leiden(Xg, res_, seed):
        A = knn_graph(Xg, a.knn).tocoo()
        g = ig.Graph(n=Xg.shape[0], edges=list(zip(A.row[A.row < A.col], A.col[A.row < A.col])))
        part = g.community_leiden(objective_function="modularity", resolution=res_, n_iterations=-1)
        lab = np.array(part.membership)
        return lab, float(g.modularity(part.membership))

    import random
    rows = []
    for res_ in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
        random.seed(a.seed)
        labs, mods = [], []
        for b in range(min(a.boots, 4) + 1):
            l_, m_ = leiden(Xs[b], res_, b)
            labs.append(l_)
            mods.append(m_)
        sizes = np.bincount(labs[0])
        big = int((sizes >= 0.01 * len(labs[0])).sum())
        nl, nm = leiden(nulls[0], res_, 0)
        r = dict(resolution=res_, k=int(len(sizes)), k_ge_1pct=big, modularity=mods[0],
                 null_k=int(nl.max() + 1), null_modularity=nm,
                 stability=mean_pair_ari([transfer(Xs[b], labs[b], P) for b in range(len(labs))]),
                 ami_classes=ami(cls, labs[0]))
        if len(sizes) >= 2:
            r.update(sep_scores(X, labs[0], 10000, rng))
            r["null_silhouette"] = sep_scores(nulls[0], nl, 10000, rng)["silhouette"]
        rows.append(r)
        log("leiden", {kk: round(v, 3) if isinstance(v, float) else v for kk, v in r.items()})
    res["methods"]["leiden"] = dict(by_res=rows)
    save()

    # ---------------- HDBSCAN ----------------
    log("hdbscan")
    rows = []
    for frac in (0.005, 0.01, 0.02, 0.05):
        mcs = int(frac * a.n)
        h = HDBSCAN(min_cluster_size=mcs).fit_predict(X)
        hn = HDBSCAN(min_cluster_size=mcs).fit_predict(nulls[0])
        sizes = np.bincount(h[h >= 0]) if (h >= 0).any() else np.array([], int)
        r = dict(min_cluster_frac=frac, n_clusters=int(len(sizes)), noise=float((h < 0).mean()),
                 largest_frac=float(sizes.max() / len(h)) if len(sizes) else 0.0,
                 null_n_clusters=int(hn.max() + 1), null_noise=float((hn < 0).mean()))
        if len(sizes) >= 2:
            keep = h >= 0
            r["ami_classes_clustered"] = ami(cls[keep], h[keep])
        rows.append(r)
        log("hdbscan", r)
    res["methods"]["hdbscan"] = dict(by_mcs=rows)
    save()

    # ---------------- DP-GMM ----------------
    log("dpgmm")
    rows = []
    for wcp in (1e-3, 1e-1, 10.0):
        bg = BayesianGaussianMixture(n_components=15, weight_concentration_prior_type="dirichlet_process",
                                     weight_concentration_prior=wcp, covariance_type="full",
                                     max_iter=500, random_state=a.seed).fit(X)
        bn = BayesianGaussianMixture(n_components=15, weight_concentration_prior_type="dirichlet_process",
                                     weight_concentration_prior=wcp, covariance_type="full",
                                     max_iter=500, random_state=a.seed).fit(nulls[0])
        lab = bg.predict(X)
        r = dict(concentration=wcp, k_eff=int((bg.weights_ > 0.02).sum()),
                 null_k_eff=int((bn.weights_ > 0.02).sum()),
                 weights=sorted(bg.weights_.round(3).tolist(), reverse=True)[:8],
                 ami_classes=ami(cls, lab))
        rows.append(r)
        log("dpgmm", r)
    res["methods"]["dpgmm"] = dict(by_conc=rows)
    save()
    log("done")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    main()
