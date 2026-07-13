#!/usr/bin/env python
"""Conformational-landscape analysis of a cryoDRGN latent space against the
CryoSPARC heterogeneous-refinement classes.

Designed for the CFTR / J264 question (John, 2026-07-06): are the CryoSPARC
classes discrete states or samples along continuous conformational coordinates,
and which classes overlap?  Produces publication-oriented figures:

  landscape_by_class.png     latent PC1-PC2 density with per-class covariance
                             ellipses, class-mean markers and the hypothesised
                             motion path drawn between class means.
  conformational_axes.png    ridgeline plots of every class along named,
                             biologically-motivated 1-D coordinates (e.g. the
                             NBD-rocking axis SC->AO and an exit-portal axis),
                             classes stacked in trajectory order.
  free_energy.png            F = -log p along PC1/PC2/PC3 with class means marked
                             (are there wells = discrete states, or one basin?).
  class_overlap.png          k-NN neighbour-class matrix (assignment overlap) and
                             Bhattacharyya coefficient (latent distribution
                             overlap) -- which classes are most confused.
  core_states.png            the same landscape + ridgeline recomputed with the
                             ablated/marginal classes removed.
  traversal_zfiles/          on-manifold PC1/PC2/PC3 traversal z-files (and
                             core-set versions) to decode into volume movies on
                             a GPU once the final training is complete.

Pure-latent analysis: runs locally on the CPU from an existing z.pkl. Example::

    ./cryodrgn-py310/Scripts/python.exe scripts/cryodrgn/cryodrgn_conformational_landscape.py \
      --z results_cryodrgn/J264_real/pilot_z10/z.49.pkl \
      --passthrough data/J264/cryosparc_P7_J264_passthrough_particles_all_classes_blob.cs \
      --cs data/J264/cryosparc_P7_J264_00062_particles_alignments3D_multi.cs \
      --protein-idx 6 7 8 9 10 11 12 13 14 --n-dummies 6 \
      --class-names SC,AC,AO,SEPD,AEPD,V-shaped,NBD1-less,NBD2-less,NBD1-less-wide \
      --order SC,AC,AO,SEPD,AEPD,V-shaped,NBD1-less,NBD2-less,NBD1-less-wide \
      --coord "NBD rocking (closed->open):SC:AO" \
      --coord "Exit-portal / dissociation:SC+AC:SEPD+AEPD" \
      --ablated NBD1-less,NBD2-less,NBD1-less-wide \
      -o results_cryodrgn/conformational_landscape/J264

For a --ind-trained latent (the D=256 final) add ``--ind .../ind_keep.pkl``.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)  # scripts/ holds gmm_pipeline
_REPO = os.path.dirname(_SCRIPTS)
for p in (_REPO, _SCRIPTS, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import gaussian_kde, multivariate_normal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

import cryodrgn_latent_gmm as clg
from cryodrgn_free_energy import free_energy_1d, basin_analysis

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 160, "font.size": 11,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
})

# A curated CFTR palette: NBD-rocking classes cool blues, exit-portal greens,
# V-shaped purple, ablated warm reds/greys. Falls back to a colormap otherwise.
CFTR_COLORS = {
    "SC": "#08519c", "AC": "#3182bd", "AO": "#6baed6",
    "SEPD": "#238b45", "AEPD": "#74c476",
    "V-shaped": "#756bb1",
    "NBD1-less": "#cb181d", "NBD2-less": "#fb6a4a", "NBD1-less-wide": "#969696",
}


# --------------------------------------------------------------------------- #
# Loading / alignment
# --------------------------------------------------------------------------- #
def ind_filter_passthrough(passthrough_cs, ind_pkl, out_npy):
    pt = np.load(passthrough_cs)
    ind = pickle.load(open(ind_pkl, "rb"))
    ind = np.sort(np.asarray(ind).ravel().astype(np.int64))
    np.save(out_npy, pt[ind])
    print(f"[ind] {len(pt)} -> {len(ind)} kept rows -> {out_npy}")
    return out_npy


def load_aligned(args):
    z = clg.load_latent(args.z)
    passthrough = args.passthrough
    if args.ind:
        os.makedirs(args.outdir, exist_ok=True)
        passthrough = ind_filter_passthrough(
            args.passthrough, args.ind, os.path.join(args.outdir, "_passthrough_kept.npy"))
    z_a, _post, hard, uid, n_prot = clg.align_z_to_posteriors(
        z, passthrough, args.cs, args.n_dummies, args.protein_idx)
    return z_a, hard


def load_membership_labels(args):
    """Overlay a fresh refinement's hard class labels onto an existing latent.
    Each --class-cs 'PROTEIN_IDX:path' contributes the uids assigned to that class;
    z rows are mapped to uids through the (training) passthrough and labelled by
    membership. Particles absent from every class file (e.g. removed/ablated) are
    dropped. Lets us ask: do a re-refinement's classes carve the SAME latent more
    cleanly than the original assignments did?"""
    z = clg.load_latent(args.z)
    uid_pass = clg.cs_uids(args.passthrough)
    if len(uid_pass) != len(z):
        m = min(len(uid_pass), len(z))
        uid_pass, z = uid_pass[:m], z[:m]
    uid_to_cls = {}
    for spec in args.class_cs:
        idx_str, path = spec.split(":", 1)
        pos = args.protein_idx.index(int(idx_str))
        for u in clg.cs_uids(path).tolist():
            uid_to_cls[int(u)] = pos
    keep_z, hard = [], []
    for i, u in enumerate(uid_pass.tolist()):
        c = uid_to_cls.get(int(u))
        if c is not None:
            keep_z.append(i)
            hard.append(c)
    keep_z = np.asarray(keep_z, dtype=int)
    hard = np.asarray(hard, dtype=int)
    print(f"[membership] matched {len(keep_z):,}/{len(z):,} latent rows to "
          f"{len(args.class_cs)} class files")
    return z[keep_z], hard


# --------------------------------------------------------------------------- #
# Coordinate helpers
# --------------------------------------------------------------------------- #
def group_centroid(zstd, hard, names, spec):
    """spec like 'SC' or 'SC+AC' -> mean standardized-latent centroid."""
    members = spec.split("+")
    idx = [names.index(m.strip()) for m in members]
    rows = np.isin(hard, idx)
    return zstd[rows].mean(0)


def named_axis(zstd, hard, names, frm, to):
    """Unit vector in standardized latent from one class(group) centroid to another."""
    v = group_centroid(zstd, hard, names, to) - group_centroid(zstd, hard, names, frm)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def traversal_zfile(z, coord, n, path, lo=2.0, hi=98.0):
    edges = np.percentile(coord, np.linspace(lo, hi, n + 1))
    pts = [z[(coord >= a) & (coord <= b)].mean(0)
           for a, b in zip(edges[:-1], edges[1:])
           if ((coord >= a) & (coord <= b)).sum() > 0]
    np.savetxt(path, np.asarray(pts))


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_landscape(pcs, hard, names, order, colors, evr, path_arrows, disp, out):
    fig, ax = plt.subplots(figsize=(9.2, 7.6))
    ax.hexbin(pcs[:, 0], pcs[:, 1], gridsize=80, cmap="Greys", mincnt=1, alpha=0.5)
    for nm in order:
        i = names.index(nm)
        m = hard == i
        if m.sum() < 10:
            continue
        mu = pcs[m, :2].mean(0)
        cov = np.cov(pcs[m, :2].T)
        vals, vecs = np.linalg.eigh(cov)
        ang = np.degrees(np.arctan2(vecs[1, np.argmax(vals)], vecs[0, np.argmax(vals)]))
        w, h = 2 * 1.25 * np.sqrt(np.maximum(vals, 1e-9))
        ax.add_patch(Ellipse(mu, w, h, angle=ang, facecolor=colors[i], edgecolor="k",
                             lw=1.0, alpha=0.30, zorder=2))
        ax.scatter(*mu, s=180, c=[colors[i]], edgecolors="k", marker="o",
                   zorder=4, label=disp[nm])
    for a, b in path_arrows:
        if a in names and b in names:
            ma = pcs[hard == names.index(a), :2].mean(0)
            mb = pcs[hard == names.index(b), :2].mean(0)
            ax.annotate("", mb, ma, zorder=3,
                        arrowprops=dict(arrowstyle="-|>", color="0.2", lw=2.2,
                                        connectionstyle="arc3,rad=0.12"))
    ax.set_xlabel(f"PC1  ({evr[0]*100:.1f}% of latent variance)")
    ax.set_ylabel(f"PC2  ({evr[1]*100:.1f}%)")
    ax.set_title("Conformational landscape on PC1-PC2, coloured by CryoSPARC class")
    ax.legend(loc="upper left", fontsize=8.5, ncol=2, title="class (P#)",
              title_fontsize=9)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def ridgeline_panel(ax, coord, hard, names, panel_order, colors, disp, xlabel, title):
    """Stacked per-class density along one coordinate; classes labelled + colour-
    coded, dashed line = class mean. panel_order sets top->bottom (usually sorted
    by class mean on this axis to make the separation obvious)."""
    grid = np.linspace(np.percentile(coord, 0.5), np.percentile(coord, 99.5), 400)
    yticks, yticklabels, ytcolors = [], [], []
    for row, nm in enumerate(panel_order):
        i = names.index(nm)
        xi = coord[hard == i]
        d = gaussian_kde(xi)(grid); d = 0.92 * d / d.max()
        base = (len(panel_order) - row) * 1.0
        ax.fill_between(grid, base, base + d, color=colors[i], alpha=0.9, lw=0, zorder=2)
        ax.plot(grid, base + d, color="k", lw=0.7, zorder=3)
        mu = xi.mean()
        ax.plot([mu, mu], [base, base + 0.92], color="k", lw=1.1, ls=(0, (2, 2)),
                alpha=0.55, zorder=4)
        yticks.append(base + 0.35); yticklabels.append(disp[nm]); ytcolors.append(colors[i])
    ax.set_yticks(yticks); ax.set_yticklabels(yticklabels, fontsize=9)
    for lbl, c in zip(ax.get_yticklabels(), ytcolors):
        lbl.set_color(c); lbl.set_fontweight("bold")
    ax.set_xlabel(xlabel); ax.set_title(title); ax.margins(y=0.02)


def fig_pc_separation(pcs, hard, names, colors, disp, evr, out):
    fig, ax = plt.subplots(1, 3, figsize=(17, 6.8))
    for k in range(3):
        x = pcs[:, k]
        present = [nm for nm in names if (hard == names.index(nm)).sum() >= 20]
        panel_order = sorted(present, key=lambda nm: -x[hard == names.index(nm)].mean())
        ridgeline_panel(ax[k], x, hard, names, panel_order, colors, disp,
                        xlabel=f"PC{k+1}", title=f"PC{k+1}  ({evr[k]*100:.1f}% of variance)")
    fig.suptitle("Class density along PC1 / PC2 / PC3; classes ordered by their "
                 "position on each axis", fontweight="bold")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_free_energy(pcs, hard, names, order, colors, out, barrier_kt=0.5):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    info = {}
    for k in range(3):
        x = pcs[:, k]
        grid = np.linspace(np.percentile(x, 0.5), np.percentile(x, 99.5), 400)
        F, _ = free_energy_1d(x, grid)
        minima, barriers = basin_analysis(F, grid, barrier_kt=barrier_kt)
        ax[k].plot(grid, F, "k-", lw=2.2, zorder=3)
        ax[k].fill_between(grid, F, F.max(), color="0.93", zorder=0)
        for mi in minima:
            ax[k].plot(grid[mi], F[mi], "o", c="tab:blue", ms=9, zorder=4)
        for nm in order:
            i = names.index(nm)
            xm = x[hard == i].mean()
            ax[k].plot(xm, -0.05, marker="^", c=colors[i], ms=9, clip_on=False, zorder=5)
        ax[k].set_xlabel(f"PC{k+1}"); ax[k].set_ylabel("F = -log p  (kT)")
        ax[k].set_title(f"F(PC{k+1}): {len(minima)} well" + ("s" if len(minima) != 1 else ""))
        info[f"PC{k+1}"] = {"n_wells": len(minima),
                            "barriers_kt": [round(b['depth_from_shallower_well_kt'], 2)
                                            for b in barriers]}
    fig.suptitle("Free-energy landscape along each PC  (triangles = class means)",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return info


def class_overlap(zstd, hard, names, order, colors, disp, out, sample=40000, k=30, seed=0):
    rng = np.random.default_rng(seed)
    n = len(zstd); K = len(names)
    idx = rng.choice(n, size=min(sample, n), replace=False)
    Z, H = zstd[idx], hard[idx]
    # --- k-NN neighbour-class fraction (local overlap) ---
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    _, nbr = nn.kneighbors(Z); nbr = nbr[:, 1:]
    KNN = np.zeros((K, K))
    for i in range(K):
        rows = np.where(H == i)[0]
        if len(rows) == 0:
            continue
        lab = H[nbr[rows]]
        counts = np.array([(lab == j).sum() for j in range(K)], float)
        KNN[i] = counts / counts.sum()
    # --- GMM soft-assignment confusion (one Gaussian per class) ---
    means = [zstd[hard == i].mean(0) for i in range(K)]
    covs = [np.cov(zstd[hard == i].T) + 1e-3 * np.eye(zstd.shape[1]) for i in range(K)]
    priors = np.array([(hard == i).mean() for i in range(K)])
    logp = np.zeros((len(Z), K))
    for j in range(K):
        logp[:, j] = multivariate_normal.logpdf(Z, means[j], covs[j],
                                                allow_singular=True) + np.log(priors[j] + 1e-12)
    logp -= logp.max(1, keepdims=True)
    post = np.exp(logp); post /= post.sum(1, keepdims=True)
    GMM = np.zeros((K, K))
    for i in range(K):
        rows = H == i
        if rows.sum():
            GMM[i] = post[rows].mean(0)
    perm = [names.index(nm) for nm in order]
    onames = [disp[nm] for nm in order]
    KNNo, GMMo = KNN[np.ix_(perm, perm)], GMM[np.ix_(perm, perm)]
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.6))
    for a, Mx, ttl in ((ax[0], KNNo, "k-NN neighbour-class fraction\n(local: fraction of a "
                        "particle's neighbours in each class)"),
                       (ax[1], GMMo, "GMM soft-assignment confusion\n(global: mean per-class "
                        "Gaussian posterior)")):
        im = a.imshow(Mx, cmap="magma", vmin=0, vmax=1)
        a.set_xticks(range(len(onames))); a.set_yticks(range(len(onames)))
        a.set_xticklabels(onames, rotation=45, ha="right", fontsize=8.5)
        a.set_yticklabels(onames, fontsize=8.5)
        a.set_xlabel("assigned / neighbour class"); a.set_ylabel("true CryoSPARC class")
        a.set_title(ttl)
        for i in range(len(onames)):
            for j in range(len(onames)):
                a.text(j, i, f"{Mx[i,j]:.2f}", ha="center", va="center",
                       color="w" if Mx[i, j] < 0.55 else "k", fontsize=7)
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle("Class overlap / assignment uncertainty in latent space  "
                 "(diagonal = self-consistency)", fontweight="bold")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    confused = {}
    for i, nm in enumerate(names):
        row = KNN[i].copy(); row[i] = -1; j = int(np.argmax(row))
        confused[nm] = {"partner": names[j], "neighbour_frac": round(float(KNN[i, j]), 3),
                        "self_frac_knn": round(float(KNN[i, i]), 3),
                        "self_frac_gmm": round(float(GMM[i, i]), 3)}
    return confused


def fig_core_states(zstd, hard, names, order, colors, disp, keep, path_arrows, out):
    Zs, Hs = zstd[keep], hard[keep]
    pca = PCA(3).fit(Zs); P = pca.transform(Zs); evr = pca.explained_variance_ratio_
    core_order = [nm for nm in order if names.index(nm) in set(Hs.tolist())]
    fig, ax = plt.subplots(1, 2, figsize=(16, 6.8))
    ax[0].hexbin(P[:, 0], P[:, 1], gridsize=65, cmap="Greys", mincnt=1, alpha=0.5)
    for nm in core_order:
        i = names.index(nm); m = Hs == i
        mu = P[m, :2].mean(0); cov = np.cov(P[m, :2].T)
        vals, vecs = np.linalg.eigh(cov)
        ang = np.degrees(np.arctan2(vecs[1, np.argmax(vals)], vecs[0, np.argmax(vals)]))
        w, h = 2 * 1.25 * np.sqrt(np.maximum(vals, 1e-9))
        ax[0].add_patch(Ellipse(mu, w, h, angle=ang, facecolor=colors[i],
                                edgecolor="k", lw=1, alpha=0.30))
        ax[0].scatter(*mu, s=150, c=[colors[i]], edgecolors="k", zorder=4, label=disp[nm])
    for a, b in path_arrows:
        if a in core_order and b in core_order:
            ma = P[Hs == names.index(a), :2].mean(0)
            mb = P[Hs == names.index(b), :2].mean(0)
            ax[0].annotate("", mb, ma, arrowprops=dict(arrowstyle="-|>", color="0.2", lw=2.2,
                           connectionstyle="arc3,rad=0.12"))
    ax[0].set_xlabel(f"PC1' ({evr[0]*100:.1f}%)"); ax[0].set_ylabel(f"PC2' ({evr[1]*100:.1f}%)")
    ax[0].set_title("Core-state landscape (ablated classes removed)")
    ax[0].legend(fontsize=8.5, ncol=2, title="class (P#)")
    x = P[:, 0]
    panel_order = sorted(core_order, key=lambda nm: -x[Hs == names.index(nm)].mean())
    ridgeline_panel(ax[1], x, Hs, names, panel_order, colors, disp,
                    xlabel="PC1'", title=f"PC1' ({evr[0]*100:.1f}% of variance)")
    fig.suptitle("Core conformational states (ablated excluded)", fontweight="bold")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return {"kept_classes": core_order, "explained_var": pca.explained_variance_ratio_[:3].tolist()}, pca, P


# --------------------------------------------------------------------------- #
def parse_coord(spec):
    label, frm, to = spec.split(":")
    return (label, frm, to)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--z", required=True)
    ap.add_argument("--passthrough", required=True)
    ap.add_argument("--cs", default="",
                    help="multi-class cs with class_posterior (posterior mode)")
    ap.add_argument("--class-cs", action="append", default=[],
                    help="'PROTEIN_IDX:path' per-class cs; overlay a fresh "
                         "refinement's hard labels on the latent (membership mode)")
    ap.add_argument("--protein-idx", type=int, nargs="+", required=True)
    ap.add_argument("--n-dummies", type=int, default=6)
    ap.add_argument("--class-names", default="")
    ap.add_argument("--order", default="", help="class draw/stack order (trajectory)")
    ap.add_argument("--coord", action="append", default=[],
                    help="named coordinate 'LABEL:FROM:TO' (FROM/TO are class names; "
                         "'A+B' = group centroid). Repeatable.")
    ap.add_argument("--path", default="",
                    help="motion arrows between class means, e.g. 'SC>AC>AO>SEPD>AEPD'")
    ap.add_argument("--ablated", default="")
    ap.add_argument("--ind", default="")
    ap.add_argument("--n-traj", type=int, default=10)
    ap.add_argument("--barrier-kt", type=float, default=0.5)
    ap.add_argument("-o", "--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    names = (args.class_names.split(",") if args.class_names
             else [f"P{i}" for i in args.protein_idx])
    assert len(names) == len(args.protein_idx)
    order = args.order.split(",") if args.order else names
    cmap = plt.get_cmap("tab10" if len(names) <= 10 else "tab20")
    colors = [CFTR_COLORS.get(nm, cmap(i % cmap.N)) for i, nm in enumerate(names)]
    coords = [parse_coord(c) for c in args.coord] or [
        ("PC-derived coordinate", order[0], order[-1])]
    if args.path:
        pth = args.path.split(">")
        path_arrows = list(zip(pth, pth[1:]))
    else:
        path_arrows = list(zip(order, order[1:]))

    z, hard = (load_membership_labels(args) if args.class_cs
               else load_aligned(args))
    zstd = StandardScaler().fit_transform(z)
    pca = PCA(3).fit(zstd); pcs = pca.transform(zstd); evr = pca.explained_variance_ratio_
    print(f"[pca] var PC1/2/3 = {evr[0]:.3f}/{evr[1]:.3f}/{evr[2]:.3f}")

    disp = {}
    for i, nm in enumerate(names):
        tag = f"P{args.protein_idx[i]}"
        disp[nm] = nm if nm == tag else f"{nm} ({tag})"

    fig_pc_separation(pcs, hard, names, colors, disp, evr,
                      os.path.join(args.outdir, "pc_separation.png"))
    fig_landscape(pcs, hard, names, order, colors, evr, path_arrows, disp,
                  os.path.join(args.outdir, "landscape_by_class.png"))
    confused = class_overlap(zstd, hard, names, order, colors, disp,
                             os.path.join(args.outdir, "class_overlap.png"))
    fe = fig_free_energy(pcs, hard, names, order, colors,
                         os.path.join(args.outdir, "free_energy.png"), args.barrier_kt)

    core = None
    if args.ablated:
        abl = [names.index(a.strip()) for a in args.ablated.split(",") if a.strip() in names]
        keep = ~np.isin(hard, abl)
        core, _, _ = fig_core_states(zstd, hard, names, order, colors, disp, keep,
                                     path_arrows, os.path.join(args.outdir, "core_states.png"))

    # traversal z-files for later GPU decoding
    zf = os.path.join(args.outdir, "traversal_zfiles"); os.makedirs(zf, exist_ok=True)
    for k in range(3):
        traversal_zfile(z, pcs[:, k], args.n_traj, os.path.join(zf, f"pc{k+1}.txt"))
    if args.ablated:
        Pz = PCA(3).fit_transform(zstd[keep])
        for k in range(3):
            traversal_zfile(z[keep], Pz[:, k], args.n_traj,
                            os.path.join(zf, f"core_pc{k+1}.txt"))

    metrics = {"n_particles": int(len(z)), "pca_explained_var": evr[:3].tolist(),
               "free_energy_wells": fe, "class_most_confused": confused, "core": core,
               "class_names": {int(args.protein_idx[i]): names[i] for i in range(len(names))}}
    json.dump(metrics, open(os.path.join(args.outdir, "landscape_metrics.json"), "w"), indent=2)

    lines = ["# Conformational-landscape analysis", "",
             f"- Particles: {len(z):,}; latent dim {z.shape[1]}",
             f"- PCA variance PC1/2/3 = {evr[0]*100:.1f}% / {evr[1]*100:.1f}% / {evr[2]*100:.1f}%",
             "", "## Discrete or continuous? (free-energy wells per PC)"]
    for pc, d in fe.items():
        lines.append(f"- F({pc}): {d['n_wells']} well(s); barriers {d['barriers_kt']} kT")
    lines += ["", "## Class self-consistency & most-confused partner "
              "(k-NN local / GMM global)"]
    for nm, d in confused.items():
        lines.append(f"- {nm}: self kNN {d['self_frac_knn']} / GMM {d['self_frac_gmm']}, "
                     f"nearest other = **{d['partner']}** ({d['neighbour_frac']})")
    if core:
        lines += ["", "## Core states (ablated excluded)",
                  f"- kept: {', '.join(core['kept_classes'])}",
                  f"- PC1'/2'/3' var: {[round(v,3) for v in core['explained_var']]}"]
    lines += ["", "Class map: " + ", ".join(f"P{args.protein_idx[i]}={names[i]}"
                                             for i in range(len(names)))]
    open(os.path.join(args.outdir, "landscape_summary.md"), "w").write("\n".join(lines) + "\n")
    print(f"[done] -> {args.outdir}")


if __name__ == "__main__":
    main()
