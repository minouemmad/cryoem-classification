#!/usr/bin/env python
"""2D free-energy basins F(PC1,PC2) + CryoSPARC-class x basin occupancy matrix.

This answers the two tests proposed after the J1442/J1497 free-energy analysis:

  TEST 1 ("the test I would absolutely perform")
      Instead of clustering the 1-D profile F(PC1), build the *2-D* free-energy
      surface  F(PC1,PC2) = -log p(PC1,PC2)  and cluster directly on it.  If the
      latent really hides extra states along PC2 (or a curved manifold) they
      show up as extra 2-D basins.  If we still recover the same small number of
      wells, we become confident the landscape genuinely has that many dominant
      basins.

  TEST 2 ("the occupancy matrix")
      For every CryoSPARC class (P6..P10) compute the fraction of its particles
      that fall in each 2-D free-energy basin:

                       Basin1  Basin2  Basin3
              P6        ...     ...     ...
              ...
              P10       ...     ...     ...

      A near-block-diagonal matrix => classes are distinct states.  Rows that
      pile two classes into the *same* basin reveal which CryoSPARC classes are
      sub-divisions of one free-energy minimum rather than independent states.

Method
------
Basins are found by topological persistence (a watershed flood of the
free-energy surface):
  * every local minimum of F is a candidate basin (its "birth" = F at the min);
  * two basins first touch at a saddle; the shallower one "dies" there;
  * its persistence = saddle_F - birth = the barrier (kT) that must be crossed
    to leave it from the shallower side -- exactly the 1-D "barrier from the
    shallower well", generalised to 2-D;
  * basins whose persistence < --barrier-kt are merged into their deeper
    neighbour (noise / corrugation), the rest survive as real basins.
This is model-free (no GMM is imposed) and gives both the basin count and a
hard basin label for every particle, from which the occupancy matrix follows.

Run with the cryoDRGN env from repo root::

    python scripts/cryodrgn/cryodrgn_basin_occupancy.py \
      --dataset "J1442:results_cryodrgn/J1442_real/train_z10/z.100.pkl:data/cryosparc_P25_J1442_passthrough_particles_all_classes.cs:data/cryosparc_P25_J1442_00000_particles.cs:6,7,8" \
      --dataset "J1497:results_cryodrgn/J1497_real/train/z.100.pkl:data/gP25W6J1497_passthrough_particles_all_classes.cs:data/cryosparc_P25_J1497_00000_particles.cs:6,7,8,9,10" \
      --n-dummies 6 --barrier-kt 0.5 -o results_cryodrgn/basin_occupancy
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
from matplotlib.colors import ListedColormap
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import cryodrgn_latent_gmm as clg


# --------------------------------------------------------------------------- #
# 2-D free energy + watershed-persistence basins
# --------------------------------------------------------------------------- #
def free_energy_2d(pc1, pc2, nx, ny, bw_scale, sub, seed):
    """F(PC1,PC2) = -log p on an (ny,nx) grid via Gaussian KDE (min shifted 0).

    Returns (F, gx, gy) where gx/gy are the grid-cell *centres*.  The KDE is fit
    on a random subsample (KDE eval is O(n_train * n_grid)) but evaluated on the
    full grid; particle assignment later uses all particles.
    """
    rng = np.random.default_rng(seed)
    xy = np.vstack([pc1, pc2])
    take = min(sub, xy.shape[1])
    fit = xy[:, rng.integers(0, xy.shape[1], size=take)]
    kde = gaussian_kde(fit, bw_method="scott")
    kde.set_bandwidth(kde.factor * bw_scale)

    gx = np.linspace(pc1.min(), pc1.max(), nx)
    gy = np.linspace(pc2.min(), pc2.max(), ny)
    GX, GY = np.meshgrid(gx, gy)
    P = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(GY.shape)
    F = -np.log(np.clip(P, 1e-300, None))
    F -= F.min()
    return F, gx, gy


_NB = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))


def _flood(F, threshold):
    """Single watershed flood of F (low = probable).

    threshold is None  -> pure merge tree: return a list of basins with
        birth (F at the local min), death (lowest saddle to a deeper basin) and
        persistence; the deepest basin has infinite persistence.
    threshold given     -> merge any basin with persistence < threshold into its
        deeper neighbour; return (labels[ny,nx], centers{label: (iy,ix)}).
    """
    ny, nx = F.shape
    flat = F.ravel()
    order = np.argsort(flat, kind="stable")
    labels = np.full(flat.size, -1, dtype=np.int64)

    parent: list[int] = []
    birth: list[float] = []       # F at the basin's local minimum
    death: list[float] = []       # F at the saddle where it merges (inf if alive)
    min_idx: list[int] = []       # flat index of the local minimum

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for idx in order:
        y, x = divmod(idx, nx)
        f = flat[idx]
        roots = set()
        for dy, dx in _NB:
            yy, xx = y + dy, x + dx
            if 0 <= yy < ny and 0 <= xx < nx:
                nidx = yy * nx + xx
                if labels[nidx] >= 0:
                    roots.add(find(labels[nidx]))
        if not roots:                                   # new local minimum
            lab = len(parent)
            parent.append(lab); birth.append(f)
            death.append(np.inf); min_idx.append(idx)
            labels[idx] = lab
        elif len(roots) == 1:
            labels[idx] = next(iter(roots))
        else:                                           # saddle: basins meet
            rl = sorted(roots, key=lambda r: birth[r])  # deepest first
            survivor = rl[0]
            for r in rl[1:]:
                if death[r] == np.inf:                  # canonical (lowest) saddle
                    death[r] = f
                pers = f - birth[r]
                if threshold is None or pers < threshold:
                    parent[r] = survivor
            labels[idx] = survivor

    if threshold is None:
        basins = []
        for i in range(len(parent)):
            d = death[i]
            pers = (d - birth[i]) if np.isfinite(d) else np.inf
            basins.append({"min_idx": int(min_idx[i]), "birth": float(birth[i]),
                           "death": (float(d) if np.isfinite(d) else None),
                           "persistence": (float(pers) if np.isfinite(pers)
                                           else None)})
        return basins

    final = np.array([find(l) if l >= 0 else -1 for l in labels])
    roots_sorted = sorted(set(final[final >= 0].tolist()))
    remap = {r: i for i, r in enumerate(roots_sorted)}
    lab2 = np.array([remap[r] if r >= 0 else -1 for r in final]).reshape(ny, nx)

    # representative minimum (lowest birth) of each surviving basin
    centers = {}
    best_birth = {}
    for i in range(len(parent)):
        r = find(i)
        if r not in best_birth or birth[i] < best_birth[r]:
            best_birth[r] = birth[i]
            centers[remap[r]] = divmod(int(min_idx[i]), nx)   # (iy, ix)
    return lab2, centers


def basin_count_curve(F, thresholds):
    """Number of surviving basins as a function of the merge threshold (kT)."""
    basins = _flood(F, None)
    pers = np.array([b["persistence"] for b in basins
                     if b["persistence"] is not None])
    # +1 for the global minimum (infinite persistence)
    return np.array([1 + int((pers >= t).sum()) for t in thresholds]), basins


def populated_basin_curve(F, gx, gy, pc1, pc2, thresholds, min_count):
    """Number of *populated* basins (>= min_count particles) vs merge threshold.

    This is the honest "how many real basins" curve: at each threshold we flood
    the surface, assign every particle, and count basins holding a meaningful
    share of the data -- the empty -log p tail basins never count."""
    out = []
    for t in thresholds:
        lab2, _ = _flood(F, t)
        pb = assign_particles(pc1, pc2, gx, gy, lab2)
        pops = np.bincount(pb, minlength=int(lab2.max()) + 1)
        out.append(int((pops >= min_count).sum()))
    return np.array(out)


# --------------------------------------------------------------------------- #
def assign_particles(pc1, pc2, gx, gy, lab2):
    """Nearest-cell basin label for every particle."""
    dx = gx[1] - gx[0]
    dy = gy[1] - gy[0]
    ix = np.clip(np.round((pc1 - gx[0]) / dx).astype(int), 0, len(gx) - 1)
    iy = np.clip(np.round((pc2 - gy[0]) / dy).astype(int), 0, len(gy) - 1)
    return lab2[iy, ix]


def merge_minor_basins(lab2, centers, gx, gy, part_basin, min_count):
    """Absorb negligibly-populated basins into the nearest real basin.

    The 2D KDE free energy has spurious local minima in the low-density tails
    (where p ~ 0, so F = -log p is large and noisy); these "basins" hold ~no
    particles and are not metastable states.  A basin survives only if it holds
    >= ``min_count`` particles; every other basin (and its grid cells) is
    re-assigned to the nearest surviving basin by latent (PC1,PC2) distance.
    Returns (lab2, centers, part_basin, n_real)."""
    n_raw = int(lab2.max()) + 1
    pops = np.bincount(part_basin, minlength=n_raw)
    real = [b for b in range(n_raw) if pops[b] >= min_count]
    if not real:
        real = [int(np.argmax(pops))]
    rc = np.array([[gx[centers[b][1]], gy[centers[b][0]]] for b in real])

    basin_map = {}
    for b in range(n_raw):
        if b in real:
            basin_map[b] = real.index(b)
        else:
            cb = np.array([gx[centers[b][1]], gy[centers[b][0]]])
            basin_map[b] = int(np.argmin(((rc - cb) ** 2).sum(1)))

    new_part = np.array([basin_map[b] for b in part_basin])
    new_lab = np.vectorize(lambda v: basin_map[v] if v >= 0 else -1)(lab2)
    new_centers = {i: centers[real[i]] for i in range(len(real))}
    return new_lab, new_centers, new_part, len(real)


def occupancy_matrices(part_basin, cryo_hard, cryo_post, n_class, n_basin):
    """Row-normalised CryoSPARC-class x basin occupancy (hard and soft)."""
    hard = np.zeros((n_class, n_basin))
    soft = np.zeros((n_class, n_basin))
    for b in range(n_basin):
        in_b = part_basin == b
        for j in range(n_class):
            hard[j, b] = np.sum(in_b & (cryo_hard == j))
            soft[j, b] = np.sum(cryo_post[in_b, j])
    hard_n = hard / np.clip(hard.sum(1, keepdims=True), 1, None)
    soft_n = soft / np.clip(soft.sum(1, keepdims=True), 1e-12, None)
    basin_pop = np.bincount(part_basin, minlength=n_basin) / len(part_basin)
    return hard_n, soft_n, basin_pop, hard.astype(int)


# --------------------------------------------------------------------------- #
def analyse(label, z_path, pass_cs, cs, protein_idx, n_dummies,
            barrier_kt, min_pop, nx, ny, bw_scale, sub, seed):
    print(f"\n=== {label} ===")
    z = clg.load_latent(z_path)
    z_a, cryo_post, cryo_hard, uid_a, n_prot = clg.align_z_to_posteriors(
        z, pass_cs, cs, n_dummies, protein_idx)

    Xs = StandardScaler().fit_transform(z_a)
    pca = PCA(n_components=2, random_state=seed).fit(Xs)
    scores = pca.transform(Xs)
    evr = pca.explained_variance_ratio_

    # orient PC1 so CryoSPARC class index increases left->right (cosmetic)
    pc1, pc2 = scores[:, 0], scores[:, 1]
    cmeans = [pc1[cryo_hard == j].mean() for j in range(len(protein_idx))]
    if np.polyfit(range(len(cmeans)), cmeans, 1)[0] < 0:
        pc1 = -pc1
    scores = np.column_stack([pc1, pc2])

    F, gx, gy = free_energy_2d(pc1, pc2, nx, ny, bw_scale, sub, seed)

    thresholds = np.round(np.arange(0.1, 3.01, 0.1), 2)
    _, basins_all = basin_count_curve(F, thresholds)

    lab2, centers = _flood(F, barrier_kt)
    n_raw = lab2.max() + 1
    part_basin = assign_particles(pc1, pc2, gx, gy, lab2)

    # drop spurious low-density tail basins (keep only those >= min_pop)
    min_count = int(min_pop * len(pc1))
    lab2, centers, part_basin, n_basin = merge_minor_basins(
        lab2, centers, gx, gy, part_basin, min_count)
    print(f"[{label}] watershed basins {n_raw} -> {n_basin} after dropping "
          f"basins with < {min_pop*100:.1f}% of particles")

    # honest "how many populated basins survive" curve vs merge threshold
    counts = populated_basin_curve(F, gx, gy, pc1, pc2, thresholds, min_count)

    # order basins left->right by PC1 of their minimum so "Basin 1" is leftmost
    order = sorted(range(n_basin), key=lambda b: gx[centers[b][1]])
    relabel = {old: new for new, old in enumerate(order)}
    lab2 = np.vectorize(lambda v: relabel[v] if v >= 0 else -1)(lab2)
    part_basin = np.array([relabel[b] for b in part_basin])
    centers = {relabel[b]: centers[b] for b in centers}
    basin_pos = [(float(gx[centers[b][1]]), float(gy[centers[b][0]]),
                  float(F[centers[b]])) for b in range(n_basin)]

    hard_n, soft_n, basin_pop, hard_counts = occupancy_matrices(
        part_basin, cryo_hard, cryo_post, len(protein_idx), n_basin)

    print(f"[{label}] PC1 {evr[0]*100:.1f}% PC2 {evr[1]*100:.1f}% | "
          f"2D basins @ {barrier_kt} kT = {n_basin} | "
          f"basin pops {np.round(basin_pop, 3).tolist()}")
    print(f"[{label}] occupancy (hard, rows=class P{protein_idx}):")
    for j, p in enumerate(protein_idx):
        print(f"    P{p}: " + "  ".join(f"B{b+1}={hard_n[j,b]:.2f}"
                                        for b in range(n_basin)))

    return {
        "label": label,
        "n_particles": int(len(pc1)),
        "zdim": int(z_a.shape[1]),
        "protein_idx": list(protein_idx),
        "pc1_explained_var": float(evr[0]),
        "pc2_explained_var": float(evr[1]),
        "barrier_kt": float(barrier_kt),
        "n_basins": int(n_basin),
        "n_basins_raw": int(n_raw),
        "min_pop": float(min_pop),
        "basin_positions_pc1_pc2_F": basin_pos,
        "basin_populations": basin_pop.tolist(),
        "basin_count_vs_threshold": {
            "threshold_kt": thresholds.tolist(),
            "n_basins": counts.tolist()},
        "persistences_kt": sorted(
            [b["persistence"] for b in basins_all
             if b["persistence"] is not None], reverse=True),
        "occupancy_hard": hard_n.tolist(),
        "occupancy_soft": soft_n.tolist(),
        "occupancy_hard_counts": hard_counts.tolist(),
        # plotting only
        "_scores": scores, "_F": F, "_gx": gx, "_gy": gy, "_lab2": lab2,
        "_centers": centers, "_part_basin": part_basin, "_cryo_hard": cryo_hard,
        "_thresholds": thresholds, "_counts": counts,
        "_hard_n": hard_n, "_soft_n": soft_n,
        "_uid_a": uid_a, "_pass_cs": pass_cs,
    }


# --------------------------------------------------------------------------- #
# Hybrid pipeline: per-basin particle export + basin->substate plan
# --------------------------------------------------------------------------- #
def basin_substate_plan(res):
    """Derive, straight from the occupancy matrix, which CryoSPARC classes share
    each free-energy basin and therefore which to try to re-separate (and with
    what K) by a *within-basin* hetero-refinement.

    A class is assigned to the basin where most of its particles live (the argmax
    of its occupancy row).  Classes that share a basin are candidate *structural
    substates* of one *energetic* state; the recommended hetero-refine K for that
    basin is simply how many classes pile into it (K=1 => the basin already is a
    single CryoSPARC class, no substate split to test).
    """
    hard = np.array(res["occupancy_hard"])          # [n_class, n_basin]
    protein_idx = res["protein_idx"]
    n_basin = res["n_basins"]
    dom = hard.argmax(1)                             # dominant basin per class
    plan = []
    for b in range(n_basin):
        members = [j for j in range(len(protein_idx)) if dom[j] == b]
        plan.append({
            "basin": b + 1,
            "population": float(res["basin_populations"][b]),
            "classes": [protein_idx[j] for j in members],
            "occupancy": [float(hard[j, b]) for j in members],
            "recommended_K": max(len(members), 1),
        })
    return plan


def export_basins(res, outdir):
    """Write one CryoSPARC-importable .cs per basin (subset of the passthrough by
    uid) + a per-particle (uid, basin, CryoSPARC-class) assignment CSV.

    These are the inputs for the hybrid pipeline's CryoSPARC leg:
      basin_N_particles.cs -> Import Particle Stack -> NU-refine = energetic-state
      map; then (for multi-class basins) focused hetero-refine K=recommended_K
      to test the structural substates.
    """
    label = res["label"]
    uid_a = res["_uid_a"].astype(np.uint64)
    part_basin = res["_part_basin"]
    cryo_hard = res["_cryo_hard"]
    protein_idx = res["protein_idx"]
    n_basin = res["n_basins"]

    cs = np.load(res["_pass_cs"])
    cs_uids = cs["uid"].astype(np.uint64)
    uid_to_row = {int(u): i for i, u in enumerate(cs_uids.tolist())}

    bdir = os.path.join(outdir, "basin_particles")
    os.makedirs(bdir, exist_ok=True)

    apath = os.path.join(bdir, f"{label}_basin_assignments.csv")
    with open(apath, "w", encoding="utf-8") as f:
        f.write("uid,basin,cryosparc_hard_class\n")
        for u, b, c in zip(uid_a.tolist(), part_basin.tolist(),
                           cryo_hard.tolist()):
            f.write(f"{int(u)},{b + 1},P{protein_idx[c]}\n")
    print(f"[export] {apath}")

    counts = []
    for b in range(n_basin):
        sel = np.where(part_basin == b)[0]
        rows = [uid_to_row[int(uid_a[i])] for i in sel
                if int(uid_a[i]) in uid_to_row]
        subset = cs[np.array(rows, dtype=np.int64)]
        out = os.path.join(bdir, f"{label}_basin{b + 1}_particles.cs")
        with open(out, "wb") as fh:
            np.save(fh, subset)
        counts.append(int(len(subset)))
        print(f"[export] {out}  ({len(subset):,} particles)")
    res["basin_export_counts"] = counts
    return counts


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_dataset(res, outdir):
    label = res["label"]
    F, gx, gy, lab2 = res["_F"], res["_gx"], res["_gy"], res["_lab2"]
    centers, scores = res["_centers"], res["_scores"]
    part_basin, cryo_hard = res["_part_basin"], res["_cryo_hard"]
    protein_idx = res["protein_idx"]
    n_basin = res["n_basins"]
    class_names = [f"P{j}" for j in protein_idx]
    GX, GY = np.meshgrid(gx, gy)

    basin_cmap = ListedColormap(plt.cm.tab10(np.arange(max(n_basin, 1))))
    class_cmap = plt.cm.Set1(np.linspace(0, 1, max(len(class_names), 3)))

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axF, axB, axO, axC = axes.ravel()

    # (A) 2D free-energy surface with basin minima
    Fp = np.clip(F, 0, 6)
    cf = axF.contourf(GX, GY, Fp, levels=20, cmap="viridis_r")
    axF.contour(GX, GY, Fp, levels=10, colors="k", linewidths=0.3, alpha=0.35)
    for b in range(n_basin):
        iy, ix = centers[b]
        axF.plot(gx[ix], gy[iy], "*", color="white", ms=20,
                 markeredgecolor="black", markeredgewidth=1.2)
        axF.annotate(f"Basin {b+1}", (gx[ix], gy[iy]), color="white",
                     fontsize=11, fontweight="bold", ha="center", va="bottom",
                     xytext=(0, 12), textcoords="offset points")
    fig.colorbar(cf, ax=axF, fraction=0.046, pad=0.04, label="F / kT")
    axF.set_xlabel("PC1"); axF.set_ylabel("PC2")
    axF.set_title(f"A. 2D free energy F(PC1,PC2) -- {n_basin} basin(s) "
                  f"@ {res['barrier_kt']} kT")

    # (B) watershed basin map + particle assignment boundaries
    axB.imshow(lab2, origin="lower", extent=[gx[0], gx[-1], gy[0], gy[-1]],
               aspect="auto", cmap=basin_cmap, alpha=0.45,
               vmin=0, vmax=max(n_basin - 1, 1))
    axB.contour(GX, GY, lab2, levels=np.arange(0.5, n_basin), colors="k",
                linewidths=1.0)
    for b in range(n_basin):
        iy, ix = centers[b]
        axB.plot(gx[ix], gy[iy], "*", color="white", ms=18,
                 markeredgecolor="black", markeredgewidth=1.2)
    axB.set_xlabel("PC1"); axB.set_ylabel("PC2")
    axB.set_title("B. Watershed basins (black = barriers / boundaries)")

    # (C) occupancy matrix heatmap (hard)
    hard_n = res["_hard_n"]
    im = axO.imshow(hard_n, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axO.set_xticks(range(n_basin))
    axO.set_xticklabels([f"Basin {b+1}" for b in range(n_basin)])
    axO.set_yticks(range(len(class_names)))
    axO.set_yticklabels(class_names)
    for j in range(len(class_names)):
        for b in range(n_basin):
            v = hard_n[j, b]
            axO.text(b, j, f"{v:.2f}", ha="center", va="center",
                     color="white" if v > 0.5 else "black", fontsize=11,
                     fontweight="bold")
    fig.colorbar(im, ax=axO, fraction=0.046, pad=0.04,
                 label="fraction of class in basin")
    axO.set_title("C. Occupancy matrix  P(basin | CryoSPARC class)")

    # (D) basin-count vs merge threshold + scatter inset legend
    axC2 = axC
    axC2.step(res["_thresholds"], res["_counts"], where="mid", color="crimson",
              lw=2.2)
    axC2.axvline(res["barrier_kt"], color="gray", ls=":", lw=1.5,
                 label=f"chosen {res['barrier_kt']} kT")
    axC2.axhline(res["n_basins"], color="gray", ls="--", lw=0.8)
    axC2.set_xlabel("merge barrier threshold (kT)")
    axC2.set_ylabel("number of populated 2D basins")
    axC2.set_ylim(0, max(res["_counts"].max() + 1, 4))
    axC2.set_title("D. Populated-basin count vs barrier threshold")
    axC2.legend(fontsize=9)

    fig.suptitle(
        f"{label}: 2D free-energy basins + occupancy  |  {res['n_particles']:,} "
        f"particles  |  PC1 {res['pc1_explained_var']*100:.1f}% / "
        f"PC2 {res['pc2_explained_var']*100:.1f}% var  |  "
        f"{n_basin} basin(s) vs {len(class_names)} CryoSPARC classes",
        fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(outdir, f"basin_occupancy_{label}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")

    # standalone scatter coloured by basin (the "do you see 5 islands?" view)
    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6))
    for b in range(n_basin):
        m = part_basin == b
        a1.scatter(scores[m, 0], scores[m, 1], s=2, alpha=0.2,
                   color=basin_cmap(b), label=f"Basin {b+1}", rasterized=True)
    a1.set_xlabel("PC1"); a1.set_ylabel("PC2")
    a1.set_title("Latent plane coloured by free-energy basin")
    lg = a1.legend(markerscale=6, fontsize=9)
    for h in lg.legend_handles:
        h.set_alpha(1)
    for j, name in enumerate(class_names):
        m = cryo_hard == j
        a2.scatter(scores[m, 0], scores[m, 1], s=2, alpha=0.2,
                   color=class_cmap[j], label=name, rasterized=True)
    a2.set_xlabel("PC1"); a2.set_ylabel("PC2")
    a2.set_title("Latent plane coloured by CryoSPARC class")
    lg = a2.legend(markerscale=6, fontsize=9)
    for h in lg.legend_handles:
        h.set_alpha(1)
    fig2.suptitle(f"{label}: basins vs CryoSPARC classes on the latent plane",
                  fontsize=13)
    fig2.tight_layout(rect=(0, 0, 1, 0.95))
    out2 = os.path.join(outdir, f"basin_scatter_{label}.png")
    fig2.savefig(out2, dpi=150)
    plt.close(fig2)
    print(f"[plot] {out2}")


def write_outputs(results, outdir):
    # per-dataset occupancy CSVs
    for res in results:
        protein_idx = res["protein_idx"]
        n_basin = res["n_basins"]
        for kind, mat in (("hard", res["occupancy_hard"]),
                          ("soft", res["occupancy_soft"])):
            path = os.path.join(outdir,
                                f"occupancy_{res['label']}_{kind}.csv")
            with open(path, "w", encoding="utf-8") as f:
                f.write("class," + ",".join(f"Basin{b+1}"
                                            for b in range(n_basin)) + "\n")
                for j, p in enumerate(protein_idx):
                    f.write(f"P{p}," + ",".join(f"{mat[j][b]:.4f}"
                            for b in range(n_basin)) + "\n")
            print(f"[csv] {path}")

    lines = ["# cryoDRGN 2D free-energy basins + occupancy matrix\n",
             "2D free energy `F(PC1,PC2) = -log p` clustered by watershed "
             "persistence (a basin survives if its barrier from the shallower "
             "side exceeds `--barrier-kt`). The occupancy matrix gives "
             "`P(basin | CryoSPARC class)` -- a near-identity matrix means the "
             "classes are distinct states; rows sharing a basin reveal classes "
             "that are sub-divisions of one free-energy minimum.\n"]
    for res in results:
        protein_idx = res["protein_idx"]
        n_basin = res["n_basins"]
        lines.append(f"## {res['label']} ({len(protein_idx)}-class, "
                     f"{res['n_particles']:,} particles)\n")
        lines.append(f"- PC1 {res['pc1_explained_var']*100:.1f}% / PC2 "
                     f"{res['pc2_explained_var']*100:.1f}% of standardized "
                     f"latent variance.")
        lines.append(f"- **2D basins @ {res['barrier_kt']} kT: "
                     f"{res['n_basins']}** (raw watershed found "
                     f"{res['n_basins_raw']}; tail basins < "
                     f"{res['min_pop']*100:.0f}% absorbed). "
                     f"1-D F(PC1) reported 3 wells for both datasets.")
        pos = ", ".join(f"B{b+1}=(PC1 {p[0]:.2f}, PC2 {p[1]:.2f})"
                        for b, p in enumerate(res["basin_positions_pc1_pc2_F"]))
        lines.append(f"- basin minima: {pos}.")
        lines.append(f"- basin populations: "
                     f"{[round(x,3) for x in res['basin_populations']]}.")
        top = ", ".join(f"{x:.2f}" for x in res["persistences_kt"][:6])
        lines.append(f"- top basin persistences (kT): {top}.")
        lines.append("\n  Occupancy `P(basin | class)` (hard):\n")
        header = "  | class | " + " | ".join(f"Basin {b+1}"
                                              for b in range(n_basin)) + " |"
        sep = "  |" + "---|" * (n_basin + 1)
        lines.append(header)
        lines.append(sep)
        for j, p in enumerate(protein_idx):
            row = "  | P{} | ".format(p) + " | ".join(
                f"{res['occupancy_hard'][j][b]:.2f}" for b in range(n_basin)) \
                + " |"
            lines.append(row)
        lines.append("")

        # --- hybrid-pipeline plan derived from this occupancy matrix --------- #
        plan = basin_substate_plan(res)
        lines.append(f"\n  ### Hybrid-pipeline plan ({res['label']})\n")
        lines.append("  Each basin = one **energetic state** (NU-refine its "
                     "particle subset). CryoSPARC classes that pile into the "
                     "*same* basin are candidate **structural substates** of "
                     "that one energetic state; the within-basin hetero-refine "
                     "`K` is just how many classes share the basin "
                     "(`K=1` => the basin already is a single class).\n")
        lines.append("  | basin | population | CryoSPARC classes in basin | "
                     "within-basin hetero-refine K |")
        lines.append("  |---|---|---|---|")
        for pl in plan:
            cls = ", ".join(f"P{c} ({o:.2f})"
                            for c, o in zip(pl["classes"], pl["occupancy"])) \
                  or "(none peak here)"
            kdesc = (f"**K={pl['recommended_K']}**"
                     if pl["recommended_K"] > 1 else
                     "K=1 (already one class; no split to test)")
            lines.append(f"  | Basin {pl['basin']} | {pl['population']:.3f} | "
                         f"{cls} | {kdesc} |")
        lines.append("")
        # narrative recommendation
        multi = [pl for pl in plan if pl["recommended_K"] > 1]
        if multi:
            recs = "; ".join(
                f"Basin {pl['basin']}: heteroref K={pl['recommended_K']} on "
                + "+".join(f"P{c}" for c in pl["classes"])
                for pl in multi)
            lines.append(f"  **Which classes to pair / what K:** {recs}.")
            lines.append("  The pairing is *not* chosen by hand -- it is read "
                         "off the occupancy matrix: the classes listed in a "
                         "basin row above are exactly the ones that share that "
                         "free-energy minimum, so they are the substates to try "
                         "to re-separate *within that basin only*.\n")
        else:
            lines.append("  **Which classes to pair / what K:** every basin "
                         "holds a single CryoSPARC class (block-diagonal "
                         "occupancy), so the basins already recover the classes "
                         "directly -- no within-basin split is warranted "
                         "(running heteroref K>1 here would impose the answer).\n")

        # exact maps to compare ------------------------------------------------ #
        lines.append("  ### Exact maps to compare\n")
        lines.append("  Independent CryoSPARC refinements are NOT in a common "
                     "frame -- rigid-body align before every comparison "
                     "(`scripts/cryodrgn/cryodrgn_focused_analysis.py` aligns "
                     "class1->class0; for basin maps use ChimeraX fitmap).\n")
        lines.append("  1. **Energetic-state maps (between basins):** NU-refine "
                     "each `basin_N_particles.cs`, then compare the basin maps "
                     "pairwise (FSC, masked CC, difference map). These answer "
                     "*are the free-energy basins genuinely distinct states?*")
        if multi:
            for pl in multi:
                names = "/".join(f"P{c}-like" for c in pl["classes"])
                lines.append(
                    f"  2. **Substate maps (within Basin {pl['basin']}):** "
                    f"focused hetero-refine K={pl['recommended_K']} on "
                    f"`basin{pl['basin']}_particles.cs` -> {names}; NU-refine "
                    f"each and compare them to each other (FSC, local "
                    f"resolution, CC, difference map, occupancies). These "
                    f"answer *do {'+'.join('P'+str(c) for c in pl['classes'])} "
                    f"survive as substates from images alone?*")
        lines.append("  3. **Cross-pipeline (the headline):** compare each "
                     "hybrid substate map to the ORIGINAL CryoSPARC hetero-"
                     "refine K=5 map of the same class (e.g. Basin-1 substate "
                     "vs original P6 and P10) via a CC matrix. If the hybrid "
                     "reproduces the 5 original maps, you have shown "
                     "`5 reconstructable maps = {} basins x substates` -- "
                     "structural heterogeneity nested inside fewer energetic "
                     "states.".format(res['n_basins']))
        lines.append("")

    out = os.path.join(outdir, "basin_occupancy_summary.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[summary] {out}")

    js = [{k: v for k, v in r.items() if not k.startswith("_")}
          for r in results]
    with open(os.path.join(outdir, "basin_occupancy_metrics.json"), "w",
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
    ap.add_argument("--barrier-kt", type=float, default=0.5,
                    help="minimum 2D barrier (persistence, kT) for a basin to "
                         "survive merging. 0.5 matches the 1-D F(PC1) merge.")
    ap.add_argument("--min-pop", type=float, default=0.01,
                    help="minimum particle fraction for a basin to be kept; "
                         "smaller tail basins are absorbed into the nearest "
                         "real basin (removes -log p tail artefacts).")
    ap.add_argument("--grid", type=int, default=140,
                    help="grid resolution per axis for the 2D KDE/watershed.")
    ap.add_argument("--bw-scale", type=float, default=1.0,
                    help="KDE bandwidth multiplier (Scott's rule x this).")
    ap.add_argument("--sub", type=int, default=40000,
                    help="subsample size for fitting the 2D KDE.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--export-basins", action="store_true",
                    help="write one CryoSPARC-importable .cs per basin (subset "
                         "of the passthrough) + a uid->basin assignment CSV, "
                         "for the hybrid energetic-state / substate pipeline.")
    ap.add_argument("-o", "--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    results = []
    for spec in args.dataset:
        label, z_path, pass_cs, cs, idx = spec.split(":")
        protein_idx = [int(t) for t in idx.split(",")]
        res = analyse(label, z_path, pass_cs, cs, protein_idx, args.n_dummies,
                      args.barrier_kt, args.min_pop, args.grid, args.grid,
                      args.bw_scale, args.sub, args.seed)
        plot_dataset(res, args.outdir)
        if args.export_basins:
            export_basins(res, args.outdir)
        results.append(res)

    write_outputs(results, args.outdir)
    print(f"\n[done] 2D basins + occupancy -> {args.outdir}")


if __name__ == "__main__":
    main()
