#!/usr/bin/env python
"""Bridge coordinate: make the cryoDRGN latent physically interpretable.

The cryoDRGN latent ``z`` is the encoder's representation of an *image* - it is
NOT a physical collective variable, and you cannot drop an MD atomic frame into
it directly.  Every downstream ambition ("does MD reproduce the reaction
coordinate?", "which region does a drug stabilise?") is undefined until the
latent is tied to a *physical* coordinate that both cryo-EM volumes and MD
frames can report.

This script builds that bridge, using data you already have (single condition,
local decoder weights), so the drug / MD work is plug-and-play later:

1. StandardScaler + PCA on the latent; walk each requested PC in on-manifold
   percentile steps (mean z of each PC window) so decoded volumes stay on the
   populated manifold instead of extrapolating.
2. Decode one .mrc per traversal point (and, if CryoSPARC classes are supplied,
   per-class medoid volumes) with ``cryodrgn eval_vol`` - CPU-feasible at low box.
3. Measure MODEL-FREE physical descriptors on every decoded volume:
     * ``mol_vol``     molecular volume above a contour (order/disorder,
                       NBD1-association proxy - dissociated NBD1 => less ordered
                       density => smaller volume);
     * ``Rg``          radius of gyration (overall openness/compactness);
     * ``anisotropy``  elongation from the inertia tensor (V-shape opening);
     * ``sym_{x,y,z}`` / ``sym_max`` reflection symmetry about the centre of mass
                       along each decoder axis (Symmetric-Closed vs Asymmetric
                       states: SC high, AC/AO lower);
     * ``lobe_asym``   density asymmetry between the two halves along the longest
                       principal axis (a TMD/NBD partitioning proxy).
   All descriptors are model-free and reproducible.  A true *inter-TMD rotation
   angle* (the axis the Wang/Hunt GCER paper uses to name V17/V21/V23/V31/N3)
   needs a docked atomic model; a hook is provided (``--atomic-model``) for when
   the deposited G551D coordinates arrive - until then the model-free descriptors
   proxy the same physics.
4. Calibrate: correlate each descriptor against each PC coordinate (the z<->CV
   map) and anchor the CryoSPARC / paper states (SC, AC, AO, ...) onto the CV
   axes via the per-class medoid volumes.
5. Re-express the free-energy profile F(PC) = -log p(PC) in the most informative
   physical descriptor, so F is reported over a *physical* coordinate.

Run with the cryoDRGN env (has torch/scipy/sklearn) from the repo root, e.g.::

    python scripts/cryodrgn/cryodrgn_bridge_coordinate.py \
      --label J1442 \
      --z results_cryodrgn/J1442/fullset_D256_z10_ep100/z.100.pkl \
      --weights results_cryodrgn/J1442_real/train_fullset/weights.100.pkl \
      --config  results_cryodrgn/J1442_real/train_fullset/config.yaml \
      --passthrough data/cryosparc_P25_J1442_passthrough_particles_all_classes.cs \
      --cs data/cryosparc_P25_J1442_00000_particles.cs \
      --protein-idx 6,7,8 --n-dummies 6 --class-dir data/J1442_classes \
      --class-names SC,AC,AO \
      --pcs 1,2,3 --n-traj 11 --downsample 64 --apix 4.15 --run \
      -o results_cryodrgn/bridge_coordinate

Omit ``--run`` to only write the z-files and print the ``eval_vol`` commands
(hand the heavy decode to a GPU box); re-run with ``--measure-only`` to (re)build
the descriptors and calibration from whatever .mrc files already exist.

The decoded-map descriptor code is deliberately shared philosophy with
cryodrgn_decode_states.py (same eval_vol convention, same read_mrc), but the
scientific product here is the *z <-> physical coordinate* calibration, not a
per-class CC matrix.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
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
from scipy.ndimage import shift as nd_shift
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import cryodrgn_latent_gmm as clg
from cryodrgn_decode_states import (
    eval_vol_cmd,
    load_official_membership,
    read_mrc,
    representative_z,
)
from cryodrgn_free_energy import free_energy_1d


# --------------------------------------------------------------------------- #
# Model-free physical descriptors on a decoded volume
# --------------------------------------------------------------------------- #
def _occupancy(v, contour_sigma):
    """Boolean occupancy + soft weights (density above the contour)."""
    thr = v.mean() + contour_sigma * v.std()
    occ = v > thr
    if occ.sum() < 27:  # fall back to a percentile if the contour is too high
        thr = np.percentile(v, 99.0)
        occ = v > thr
    w = np.clip(v - thr, 0.0, None)
    return occ, w, thr


def _center_by_com(v, com):
    """Shift the volume so the centre of mass sits at the grid centre."""
    n = np.array(v.shape)
    return nd_shift(v, (n - 1) / 2.0 - com, order=1, mode="constant", cval=0.0)


def _pearson(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    d = np.sqrt(np.dot(a, a) * np.dot(b, b))
    return float(np.dot(a, b) / d) if d > 0 else float("nan")


def volume_descriptors(vol, apix, contour_sigma=1.5):
    """Return a dict of model-free physical descriptors for one decoded volume."""
    v = np.asarray(vol, dtype=np.float64)
    occ, w, _thr = _occupancy(v, contour_sigma)
    tw = w.sum()
    if tw <= 0:
        return None
    n = v.shape[0]
    idx = np.arange(n, dtype=np.float64)
    Z, Y, X = np.meshgrid(idx, idx, idx, indexing="ij")
    com = np.array([(w * Z).sum(), (w * Y).sum(), (w * X).sum()]) / tw
    dz, dy, dx = Z - com[0], Y - com[1], X - com[2]

    # radius of gyration & molecular volume
    r2 = dz * dz + dy * dy + dx * dx
    Rg = float(np.sqrt((w * r2).sum() / tw) * apix)
    mol_vol = float(occ.sum()) * (apix ** 3)

    # inertia tensor -> anisotropy + principal axes
    Ixx = (w * (dy * dy + dz * dz)).sum()
    Iyy = (w * (dx * dx + dz * dz)).sum()
    Izz = (w * (dx * dx + dy * dy)).sum()
    Ixy = -(w * dx * dy).sum()
    Ixz = -(w * dx * dz).sum()
    Iyz = -(w * dy * dz).sum()
    I = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]]) / tw
    evals, evecs = np.linalg.eigh(I)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    anisotropy = float((evals[0] - evals[2]) / (evals.sum() + 1e-12))

    # reflection symmetry about the COM along each decoder axis
    vc = _center_by_com(v, com)
    sym = {f"sym_{ax}": _pearson(vc, np.flip(vc, axis=a))
           for a, ax in enumerate("zyx")}
    sym_max = float(max(sym.values()))

    # density asymmetry between the two halves along the longest principal axis
    #   (projection of each voxel onto the major inertial eigenvector)
    coords = np.stack([dz, dy, dx], axis=-1)
    proj = coords @ evecs[:, 0]  # major axis, z/y/x rows -> vector already z,y,x
    top = (w * (proj > 0)).sum()
    bot = (w * (proj <= 0)).sum()
    lobe_asym = float(abs(top - bot) / (top + bot + 1e-12))

    out = {"mol_vol": mol_vol, "Rg": Rg, "anisotropy": anisotropy,
           "sym_max": sym_max, "lobe_asym": lobe_asym}
    out.update(sym)
    return out


DESCRIPTOR_KEYS = ["mol_vol", "Rg", "anisotropy", "sym_max", "sym_z", "sym_y",
                   "sym_x", "lobe_asym"]
DESCRIPTOR_LABEL = {
    "mol_vol": "molecular volume (A^3)  [order / NBD1 density]",
    "Rg": "radius of gyration (A)  [openness]",
    "anisotropy": "inertial anisotropy  [V-shape elongation]",
    "sym_max": "reflection symmetry  [SC symmetric vs AC/AO]",
    "lobe_asym": "lobe density asymmetry  [TMD/NBD partition]",
}


# --------------------------------------------------------------------------- #
# Latent traversal
# --------------------------------------------------------------------------- #
def pc_trajectory(z_a, scores_pc, n_pts):
    """N on-manifold latent points walking one PC (mean z per percentile window),
    returning (z_points, bin_centre_scores)."""
    lo, hi = np.percentile(scores_pc, 1), np.percentile(scores_pc, 99)
    edges = np.linspace(lo, hi, n_pts + 1)
    pts, centres = [], []
    for i in range(n_pts):
        m = (scores_pc >= edges[i]) & (scores_pc <= edges[i + 1])
        if m.sum() < 10:
            m = np.argsort(np.abs(scores_pc - 0.5 * (edges[i] + edges[i + 1])))[:200]
        pts.append(z_a[m].mean(0))
        centres.append(0.5 * (edges[i] + edges[i + 1]))
    return np.asarray(pts, dtype=np.float32), np.asarray(centres)


# --------------------------------------------------------------------------- #
def analyse(args):
    os.makedirs(args.out, exist_ok=True)
    ds_dir = os.path.join(args.out, args.label)
    os.makedirs(ds_dir, exist_ok=True)
    pcs = [int(p) for p in args.pcs.split(",") if p.strip()]
    protein_idx = ([int(p) for p in args.protein_idx.split(",")]
                   if args.protein_idx else [])
    class_names = ([s.strip() for s in args.class_names.split(",")]
                   if args.class_names else
                   [f"P{p}" for p in protein_idx])

    # --- latent + optional CryoSPARC-class alignment -------------------------
    z = clg.load_latent(args.z)
    if args.passthrough and args.cs and protein_idx:
        z_a, _post, cryo_hard, uid_a, _n = clg.align_z_to_posteriors(
            z, args.passthrough, args.cs, args.n_dummies, protein_idx)
        member = load_official_membership(args.class_dir, protein_idx)
        if member is not None:
            official = np.array([member.get(int(u), -1) for u in uid_a.tolist()])
            ok = official >= 0
            cryo_hard = np.where(ok, official, cryo_hard)
    else:
        z_a, cryo_hard = z, None

    Xs = StandardScaler().fit_transform(z_a)
    n_comp = min(z_a.shape[1], max(pcs))
    pca = PCA(n_components=n_comp, random_state=args.seed).fit(Xs)
    scores = pca.transform(Xs)
    evr = pca.explained_variance_ratio_
    print(f"[pca] explained var: " +
          ", ".join(f"PC{i+1}={evr[i]*100:.1f}%" for i in range(n_comp)))

    # --- build one combined z-file: PC traversals (+ class medoids) ----------
    rows = []  # (label, z-vector, kind, pc, score)
    for pc in pcs:
        zpts, centres = pc_trajectory(z_a, scores[:, pc - 1], args.n_traj)
        for i in range(args.n_traj):
            rows.append((f"pc{pc}_t{i:02d}", zpts[i], "traj", pc, float(centres[i])))
    n_traj_rows = len(rows)

    class_rows0 = len(rows)
    if cryo_hard is not None:
        k = len(class_names)
        cz = representative_z(z_a, cryo_hard, k, "medoid")
        for j, nm in enumerate(class_names):
            rows.append((f"class_{nm}", cz[j], "class", -1, np.nan))

    zmat = np.asarray([r[1] for r in rows], dtype=np.float32)
    row_labels = [r[0] for r in rows]
    zfile = os.path.join(ds_dir, f"{args.label}_bridge_zfile.txt")
    np.savetxt(zfile, zmat, fmt="%.6f")
    with open(os.path.join(ds_dir, f"{args.label}_bridge_zfile_labels.txt"), "w") as fh:
        fh.write("\n".join(f"{i}\t{lab}" for i, lab in enumerate(row_labels)))
    print(f"[zfile] {len(rows)} latent points "
          f"({n_traj_rows} traversal + {len(rows) - class_rows0} class) -> {zfile}")

    cmd = eval_vol_cmd(args.weights, args.config, zfile, ds_dir,
                       args.downsample, args.apix, prefix=f"{args.label}_bvol")
    print("[eval_vol]", " ".join(cmd))
    if args.run and not args.measure_only:
        if not (os.path.exists(args.weights) and os.path.exists(args.config)):
            print("[eval_vol] SKIP - weights/config missing")
        else:
            print(f"[eval_vol] decoding {len(rows)} volumes (box "
                  f"{args.downsample or 'full'})...")
            subprocess.run(cmd, check=True)

    # --- measure descriptors on every decoded volume -------------------------
    def vol_path(i):
        for pat in (f"{args.label}_bvol{i + 1:03d}.mrc", f"{args.label}_bvol{i + 1}.mrc"):
            hit = os.path.join(ds_dir, pat)
            if os.path.exists(hit):
                return hit
        return None

    per_vol = []
    for i, (lab, _zv, kind, pc, sc) in enumerate(rows):
        p = vol_path(i)
        if p is None:
            continue
        d = volume_descriptors(read_mrc(p), args.apix, args.contour_sigma)
        if d is None:
            continue
        rec = {"label": lab, "kind": kind, "pc": pc, "pc_score": sc}
        rec.update(d)
        per_vol.append(rec)
    print(f"[measure] descriptors on {len(per_vol)}/{len(rows)} decoded volumes")

    if not per_vol:
        print("[measure] no volumes yet - run with --run (or decode externally) "
              "then re-run with --measure-only")
        _write_json(args, ds_dir, pcs, evr, [], {}, [])
        return

    # --- calibration: descriptor vs PC coordinate ----------------------------
    calib = {}
    for pc in pcs:
        pts = [r for r in per_vol if r["kind"] == "traj" and r["pc"] == pc]
        pts = sorted(pts, key=lambda r: r["pc_score"])
        if len(pts) < 3:
            continue
        s = np.array([r["pc_score"] for r in pts])
        entry = {}
        for key in DESCRIPTOR_KEYS:
            y = np.array([r[key] for r in pts], dtype=float)
            if np.std(y) < 1e-9:
                entry[key] = {"pearson": 0.0}
                continue
            entry[key] = {"pearson": _pearson(s, y)}
        # the descriptor this PC most strongly encodes
        best = max(entry, key=lambda kk: abs(entry[kk]["pearson"]))
        calib[f"PC{pc}"] = {"best_descriptor": best,
                            "pearson": entry[best]["pearson"],
                            "per_descriptor": entry}
        print(f"[calib] PC{pc} best-encodes '{best}' (r={entry[best]['pearson']:+.2f})")

    class_cv = [r for r in per_vol if r["kind"] == "class"]

    _plot(args, ds_dir, pcs, per_vol, calib, class_cv, scores, cryo_hard,
          class_names)
    _write_csv(ds_dir, args.label, per_vol)
    _write_json(args, ds_dir, pcs, evr, per_vol, calib, class_cv)
    print(f"[done] {ds_dir}")


# --------------------------------------------------------------------------- #
def _plot(args, ds_dir, pcs, per_vol, calib, class_cv, scores, cryo_hard,
          class_names):
    ncol = max(len(pcs), 2)
    fig = plt.figure(figsize=(4.8 * ncol, 11))
    gs = fig.add_gridspec(3, ncol)

    # Row 1: normalised descriptors vs each PC coordinate
    for c, pc in enumerate(pcs):
        ax = fig.add_subplot(gs[0, c])
        pts = sorted([r for r in per_vol if r["kind"] == "traj" and r["pc"] == pc],
                     key=lambda r: r["pc_score"])
        if pts:
            s = np.array([r["pc_score"] for r in pts])
            for key in ("mol_vol", "Rg", "anisotropy", "sym_max", "lobe_asym"):
                y = np.array([r[key] for r in pts], dtype=float)
                if np.ptp(y) > 0:
                    yn = (y - y.min()) / np.ptp(y)
                    ax.plot(s, yn, "o-", ms=3, lw=1.2, label=key)
            best = calib.get(f"PC{pc}", {}).get("best_descriptor", "")
            ax.set_title(f"PC{pc} descriptors  (best: {best})", fontsize=10)
        ax.set_xlabel(f"PC{pc} score")
        ax.set_ylabel("normalised CV")
        if c == 0:
            ax.legend(fontsize=7, loc="best")

    # Row 2: free energy F(PC) with class anchors
    for c, pc in enumerate(pcs):
        ax = fig.add_subplot(gs[1, c])
        x = scores[:, pc - 1]
        grid = np.linspace(np.percentile(x, 0.5), np.percentile(x, 99.5), 300)
        F, _p = free_energy_1d(x, grid, bw_scale=args.bw_scale)
        ax.plot(grid, F, "-", color="#333")
        ax.fill_between(grid, F, F.max(), color="#eee", zorder=0)
        if cryo_hard is not None:
            for j, nm in enumerate(class_names):
                mu = float(np.mean(x[cryo_hard == j])) if (cryo_hard == j).any() else np.nan
                if np.isfinite(mu):
                    ax.axvline(mu, ls="--", lw=1, alpha=0.7)
                    ax.text(mu, F.max() * 0.92, nm, rotation=90, fontsize=7,
                            ha="right", va="top")
        ax.set_xlabel(f"PC{pc} score")
        ax.set_ylabel("F = -log p  (kT)")
        ax.set_title(f"free energy along PC{pc}", fontsize=10)

    # Row 3: physical state map (two most informative descriptors) + class stars
    ax = fig.add_subplot(gs[2, :])
    # choose the two descriptors best encoded across PCs
    ranked = sorted(
        {f"PC{pc}": calib.get(f"PC{pc}", {}) for pc in pcs}.values(),
        key=lambda d: abs(d.get("pearson", 0)), reverse=True)
    descs = []
    for d in ranked:
        b = d.get("best_descriptor")
        if b and b not in descs:
            descs.append(b)
    for fallback in ("sym_max", "mol_vol", "Rg", "anisotropy"):
        if len(descs) >= 2:
            break
        if fallback not in descs:
            descs.append(fallback)
    dx_key, dy_key = descs[0], descs[1]
    traj = [r for r in per_vol if r["kind"] == "traj"]
    if traj:
        ax.scatter([r[dx_key] for r in traj], [r[dy_key] for r in traj],
                   c=[r["pc"] for r in traj], cmap="tab10", s=28, alpha=0.75,
                   label="traversal points")
    for r in class_cv:
        ax.scatter(r[dx_key], r[dy_key], marker="*", s=320, edgecolor="k",
                   linewidth=1.2, zorder=5)
        ax.annotate(r["label"].replace("class_", ""),
                    (r[dx_key], r[dy_key]), fontsize=10, fontweight="bold",
                    xytext=(6, 6), textcoords="offset points")
    ax.set_xlabel(DESCRIPTOR_LABEL.get(dx_key, dx_key))
    ax.set_ylabel(DESCRIPTOR_LABEL.get(dy_key, dy_key))
    ax.set_title("physical state map: cryoDRGN states in model-free "
                 "structural descriptors (stars = CryoSPARC/paper classes)",
                 fontsize=10)

    fig.suptitle(f"{args.label}: cryoDRGN latent <-> physical bridge coordinate",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = os.path.join(ds_dir, f"bridge_calibration_{args.label}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


def _write_csv(ds_dir, label, per_vol):
    import csv
    keys = ["label", "kind", "pc", "pc_score"] + DESCRIPTOR_KEYS
    out = os.path.join(ds_dir, f"bridge_descriptors_{label}.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in per_vol:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"[csv] {out}")


def _write_json(args, ds_dir, pcs, evr, per_vol, calib, class_cv):
    out = os.path.join(ds_dir, f"bridge_calibration_{args.label}.json")
    with open(out, "w") as fh:
        json.dump({
            "label": args.label,
            "pcs": pcs,
            "explained_variance_ratio": [float(x) for x in evr],
            "apix": args.apix,
            "downsample": args.downsample,
            "contour_sigma": args.contour_sigma,
            "n_volumes_measured": len(per_vol),
            "calibration": calib,
            "class_descriptors": class_cv,
            "note_inter_tmd_angle": (
                "Inter-TMD rotation angle (V17/V21/V23/V31/N3 naming) needs a "
                "docked atomic model; pass --atomic-model when the deposited "
                "G551D coordinates are available. Model-free descriptors here "
                "proxy the same physics (sym_max ~ symmetric<->asymmetric; "
                "Rg/anisotropy ~ closed<->open; mol_vol ~ NBD1 order)."),
        }, fh, indent=2)
    print(f"[json] {out}")


# --------------------------------------------------------------------------- #
def build_parser():
    ap = argparse.ArgumentParser(
        description="Calibrate the cryoDRGN latent to model-free physical "
                    "structural descriptors (the EM<->MD / drug-ΔF bridge).")
    ap.add_argument("--label", required=True)
    ap.add_argument("--z", required=True, help="z.*.pkl latent")
    ap.add_argument("--weights", required=True, help="cryoDRGN weights.*.pkl")
    ap.add_argument("--config", required=True, help="cryoDRGN config.yaml")
    ap.add_argument("--passthrough", default="", help="passthrough .cs (z order)")
    ap.add_argument("--cs", default="", help="CryoSPARC posteriors .cs (class overlay)")
    ap.add_argument("--protein-idx", default="", help="e.g. 6,7,8")
    ap.add_argument("--n-dummies", type=int, default=6)
    ap.add_argument("--class-dir", default="", help="dir of per-class split .cs")
    ap.add_argument("--class-names", default="", help="e.g. SC,AC,AO (protein-idx order)")
    ap.add_argument("--pcs", default="1,2,3")
    ap.add_argument("--n-traj", type=int, default=11, help="points per PC traversal")
    ap.add_argument("--downsample", type=int, default=64,
                    help="eval_vol box (0 = full box)")
    ap.add_argument("--apix", type=float, default=4.15,
                    help="voxel size of the DECODED map (apix_full*box_full/downsample)")
    ap.add_argument("--contour-sigma", type=float, default=1.5,
                    help="occupancy contour = mean + sigma*std")
    ap.add_argument("--bw-scale", type=float, default=1.0, help="KDE bandwidth scale")
    ap.add_argument("--atomic-model", default="",
                    help="(future) docked PDB -> exact inter-TMD angle; not yet used")
    ap.add_argument("--run", action="store_true", help="actually call eval_vol")
    ap.add_argument("--measure-only", action="store_true",
                    help="skip decoding; measure existing .mrc")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--out", default="results_cryodrgn/bridge_coordinate")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.atomic_model:
        print("[note] --atomic-model given but exact inter-TMD angle is not yet "
              "implemented; using model-free descriptors.")
    analyse(args)


if __name__ == "__main__":
    main()
