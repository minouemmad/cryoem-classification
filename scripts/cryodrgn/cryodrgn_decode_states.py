#!/usr/bin/env python
"""Decode a cryoDRGN volume at each CryoSPARC-class latent centroid and compare
the maps.  This is the *structural* complement to the density-break test
(cryodrgn_class_trajectories.py): the trajectory test asks "is there an energy
barrier between two classes?"; this asks "do two classes decode to *different
density*?".

Workflow per dataset
--------------------
1. Align the latent `z` to the CryoSPARC protein classes (same logic the rest of
   the pipeline uses), validated against the per-class split files.
2. For every class build a representative latent point.  Two flavours:
     * ``mean``   - the mean z over the class particles (the class centroid);
     * ``medoid`` - the real particle z closest to that mean (always on-manifold).
3. Optionally add a smooth PC1 trajectory (``--traj N``): N points walking the
   dominant reaction coordinate, each the mean z of a PC1 percentile window, so
   the decoded series is an on-manifold morph from one end of the landscape to
   the other.
4. Write all z-vectors to ``<out>/<label>_zfile.txt`` and call
   ``cryodrgn eval_vol`` (local ``weights.100.pkl`` + ``config.yaml``) to decode
   one .mrc per row.  Decoding is CPU-feasible; use ``-d/--downsample`` (default
   64) to keep it fast - real-space cross-correlation is robust at low box.
5. Once volumes exist, compute the pairwise real-space cross-correlation (CC)
   matrix between the per-class maps.  The scientific read-out:
     * merged pairs (e.g. J1497 P6/P10, P8/P9) -> CC ~ 1.0  => same structure,
       confirming they are substates, not distinct states;
     * genuinely separated pairs (e.g. P6/P8)  -> CC noticeably < 1 => the
       hetero-refinement split corresponds to real density differences.

Run with the cryoDRGN env from the repo root::

    python scripts/cryodrgn/cryodrgn_decode_states.py \
      --dataset "J1442:results_cryodrgn/J1442_real/train_z10/z.100.pkl:data/cryosparc_P25_J1442_passthrough_particles_all_classes.cs:data/cryosparc_P25_J1442_00000_particles.cs:6,7,8:results_cryodrgn/J1442_real/train_z10/weights.100.pkl:results_cryodrgn/J1442_real/train_z10/config.yaml:data/J1442_classes" \
      --dataset "J1497:results_cryodrgn/J1497_real/train/z.100.pkl:data/gP25W6J1497_passthrough_particles_all_classes.cs:data/cryosparc_P25_J1497_00000_particles.cs:6,7,8,9,10:results_cryodrgn/J1497_real/train/weights.100.pkl:results_cryodrgn/J1497_real/train/config.yaml:data/J1497_classes" \
      --n-dummies 6 --rep medoid --traj 8 -d 64 --run \
      -o results_cryodrgn/decode_states

Omit ``--run`` to only write the z-files and print the eval_vol commands (handy
for handing the heavy decode off to a GPU box); re-run later with
``--compare-only`` to (re)build the CC matrix from whatever .mrc files exist.
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
_REPO = os.path.dirname(os.path.dirname(_HERE))
for p in (_REPO, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import cryodrgn_latent_gmm as clg

try:
    # cryoDRGN ships an MRC reader; fall back to a tiny parser if the API moved.
    from cryodrgn.mrcfile import parse_mrc as _parse_mrc
except Exception:  # pragma: no cover - version drift
    _parse_mrc = None


# --------------------------------------------------------------------------- #
def read_mrc(path):
    """Return the volume as a float32 numpy array."""
    if _parse_mrc is not None:
        arr = _parse_mrc(path)[0]
        return np.asarray(arr, dtype=np.float32)
    # minimal MRC2014 reader (mode 2 = float32), sufficient for cubic volumes
    with open(path, "rb") as fh:
        hdr = fh.read(1024)
        nx, ny, nz, mode = np.frombuffer(hdr, dtype=np.int32, count=4)
        if mode != 2:
            raise ValueError(f"{path}: unsupported MRC mode {mode}")
        data = np.frombuffer(fh.read(), dtype=np.float32, count=int(nx) * ny * nz)
    return data.reshape(int(nz), int(ny), int(nx)).astype(np.float32)


def load_official_membership(class_dir, protein_idx):
    """uid -> CryoSPARC class index (0..K-1) from the per-class split files."""
    if not class_dir or not os.path.isdir(class_dir):
        return None
    uid_to_class = {}
    for j, p in enumerate(protein_idx):
        hits = glob.glob(os.path.join(class_dir, f"*class_0{p}_*particles.cs"))
        hits = [h for h in hits if "passthrough" not in os.path.basename(h)]
        if not hits:
            hits = glob.glob(os.path.join(class_dir, f"*passthrough*class_{p}.cs"))
        if not hits:
            print(f"[member] WARNING no per-class file for P{p} in {class_dir}")
            continue
        for u in clg.cs_uids(hits[0]).tolist():
            uid_to_class[int(u)] = j
    return uid_to_class


def cross_correlation(a, b, mask=None):
    """Normalised real-space cross-correlation between two volumes."""
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    if mask is not None:
        m = mask.ravel().astype(bool)
        a, b = a[m], b[m]
    a -= a.mean()
    b -= b.mean()
    denom = np.sqrt(np.dot(a, a) * np.dot(b, b))
    return float(np.dot(a, b) / denom) if denom > 0 else float("nan")


# --------------------------------------------------------------------------- #
def representative_z(z_a, cryo_hard, k, rep):
    """Per-class representative latent point in the ORIGINAL latent space."""
    reps = []
    for j in range(k):
        zj = z_a[cryo_hard == j]
        mu = zj.mean(0)
        if rep == "medoid":
            d = np.linalg.norm(zj - mu, axis=1)
            mu = zj[int(np.argmin(d))]
        reps.append(mu)
    return np.asarray(reps, dtype=np.float32)


def pc1_trajectory_z(z_a, n_pts, seed):
    """N on-manifold latent points walking the dominant PC1 reaction coordinate.

    Each point is the mean z of particles in a PC1 percentile window, so the
    decoded series stays on the populated manifold instead of extrapolating.
    """
    Xs = StandardScaler().fit_transform(z_a)
    pc1 = PCA(n_components=1, random_state=seed).fit_transform(Xs)[:, 0]
    edges = np.linspace(np.percentile(pc1, 1), np.percentile(pc1, 99), n_pts + 1)
    pts = []
    for i in range(n_pts):
        m = (pc1 >= edges[i]) & (pc1 <= edges[i + 1])
        if m.sum() < 10:
            m = np.argsort(np.abs(pc1 - 0.5 * (edges[i] + edges[i + 1])))[:200]
        pts.append(z_a[m].mean(0))
    return np.asarray(pts, dtype=np.float32)


def eval_vol_cmd(weights, config, zfile, outdir, downsample, apix, prefix):
    exe = os.path.join(_REPO, "cryodrgn-py310", "Scripts", "cryodrgn.exe")
    if not os.path.exists(exe):
        exe = "cryodrgn"
    cmd = [exe, "eval_vol", weights, "-c", config, "-o", outdir,
           "--zfile", zfile, "--prefix", prefix, "--Apix", str(apix)]
    if downsample:
        cmd += ["-d", str(downsample)]
    return cmd


# --------------------------------------------------------------------------- #
def analyse(label, z_path, pass_cs, cs, protein_idx, weights, config, class_dir,
            n_dummies, rep, n_traj, downsample, apix, out, run, compare_only,
            seed):
    print(f"\n=== {label} ===")
    names = [f"P{p}" for p in protein_idx]
    k = len(names)
    ds_dir = os.path.join(out, label)
    os.makedirs(ds_dir, exist_ok=True)

    z = clg.load_latent(z_path)
    z_a, cryo_post, cryo_hard, uid_a, n_prot = clg.align_z_to_posteriors(
        z, pass_cs, cs, n_dummies, protein_idx)

    member = load_official_membership(class_dir, protein_idx)
    if member is not None:
        official = np.array([member.get(int(u), -1) for u in uid_a.tolist()])
        ok = official >= 0
        agree = float(np.mean(official[ok] == cryo_hard[ok])) if ok.any() else 0.0
        print(f"[member] matched {ok.sum():,}/{len(official):,} to per-class "
              f"files; argmax vs official agreement {agree*100:.1f}%")
        cryo_hard = np.where(ok, official, cryo_hard)

    # ---- build the z-file (class centroids first, then optional trajectory) --
    class_z = representative_z(z_a, cryo_hard, k, rep)
    rows = [(nm, class_z[j]) for j, nm in enumerate(names)]
    if n_traj and n_traj > 0:
        traj_z = pc1_trajectory_z(z_a, n_traj, seed)
        rows += [(f"traj{i:02d}", traj_z[i]) for i in range(n_traj)]
    row_labels = [r[0] for r in rows]
    zmat = np.asarray([r[1] for r in rows], dtype=np.float32)

    zfile = os.path.join(ds_dir, f"{label}_zfile.txt")
    np.savetxt(zfile, zmat, fmt="%.6f")
    with open(os.path.join(ds_dir, f"{label}_zfile_labels.txt"), "w") as fh:
        fh.write("\n".join(f"{i}\t{lab}" for i, lab in enumerate(row_labels)))
    print(f"[zfile] {len(rows)} latent points ({k} class centroids "
          f"[{rep}]{' + ' + str(n_traj) + ' PC1-traj' if n_traj else ''}) "
          f"-> {zfile}")

    cmd = eval_vol_cmd(weights, config, zfile, ds_dir, downsample, apix,
                       prefix=f"{label}_vol")
    print("[eval_vol]", " ".join(cmd))

    if run and not compare_only:
        if not (os.path.exists(weights) and os.path.exists(config)):
            print(f"[eval_vol] SKIP - weights/config missing for {label}")
        else:
            print(f"[eval_vol] decoding {len(rows)} volumes "
                  f"(box {downsample or 'full'})... this may take a while on CPU")
            subprocess.run(cmd, check=True)

    # ---- compare decoded class volumes (CC matrix) ---------------------------
    vols, present = [], []
    for i, lab in enumerate(row_labels[:k]):
        # cryodrgn writes <prefix>NNN.mrc with a 1-based start index (vol001=row0)
        cand = sorted(glob.glob(os.path.join(ds_dir, f"{label}_vol{i + 1:03d}.mrc")))
        cand += sorted(glob.glob(os.path.join(ds_dir, f"{label}_vol{i + 1}.mrc")))
        cand = [c for c in cand if os.path.exists(c)]
        if cand:
            vols.append(read_mrc(cand[0]))
            present.append(i)

    cc = None
    if len(present) >= 2:
        # shared soft mask from the mean of available class maps
        ref = np.mean(vols, axis=0)
        thr = ref.mean() + 0.5 * ref.std()
        mask = ref > thr
        cc = np.full((k, k), np.nan)
        for ia, va in zip(present, vols):
            for ib, vb in zip(present, vols):
                cc[ia, ib] = cross_correlation(va, vb, mask)
        print(f"[compare] CC matrix over {len(present)}/{k} class volumes "
              f"(masked, box {vols[0].shape[0]})")
        hdr = "      " + " ".join(f"{names[j]:>6}" for j in range(k))
        print(hdr)
        for ia in range(k):
            cells = " ".join(
                ("  --  " if np.isnan(cc[ia, ib]) else f"{cc[ia, ib]:6.3f}")
                for ib in range(k))
            print(f"{names[ia]:>5} {cells}")
    else:
        print(f"[compare] only {len(present)} class volume(s) found - "
              f"run with --run (or decode externally) then --compare-only")

    return {"label": label, "names": names, "n_particles": int(len(uid_a)),
            "rep": rep, "class_counts": [int((cryo_hard == j).sum())
                                         for j in range(k)],
            "zfile": zfile, "eval_vol_cmd": " ".join(cmd),
            "volumes_present": [names[i] for i in present],
            "cc_matrix": None if cc is None else cc.tolist()}


def plot_cc(res, out):
    items = [r for r in res if r.get("cc_matrix")]
    if not items:
        return
    fig, axes = plt.subplots(1, len(items), figsize=(6.2 * len(items), 5.4),
                             squeeze=False)
    for ax, r in zip(axes[0], items):
        cc = np.array(r["cc_matrix"], dtype=float)
        names = r["names"]
        im = ax.imshow(cc, cmap="viridis", vmin=np.nanmin(cc), vmax=1.0)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names)
        ax.set_yticklabels(names)
        for i in range(len(names)):
            for j in range(len(names)):
                if not np.isnan(cc[i, j]):
                    ax.text(j, i, f"{cc[i, j]:.2f}", ha="center", va="center",
                            color="white" if cc[i, j] < 0.85 else "black",
                            fontsize=9)
        ax.set_title(f"{r['label']}: per-class decoded-map CC\n"
                     "high = same structure (substate), low = real difference")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="real-space CC")
    fig.tight_layout()
    p = os.path.join(out, "decode_states_cc.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[plot] {p}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", action="append", required=True,
                    help="LABEL:Z_PKL:PASSTHROUGH_CS:CS:PROT_IDX:WEIGHTS:CONFIG[:CLASS_DIR]")
    ap.add_argument("--n-dummies", type=int, default=6)
    ap.add_argument("--rep", choices=["mean", "medoid"], default="medoid",
                    help="per-class representative latent point (default medoid)")
    ap.add_argument("--traj", type=int, default=0,
                    help="also decode N points along the PC1 reaction coordinate")
    ap.add_argument("-d", "--downsample", type=int, default=64,
                    help="decode at this box size for speed (default 64; 0=full)")
    ap.add_argument("--apix", type=float, default=1.0,
                    help="pixel size written to .mrc header (cosmetic for CC)")
    ap.add_argument("--run", action="store_true",
                    help="actually call cryodrgn eval_vol (CPU, slow)")
    ap.add_argument("--compare-only", action="store_true",
                    help="skip decoding, only (re)build the CC matrix from .mrc")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--out", default="results_cryodrgn/decode_states")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    res = []
    for spec in args.dataset:
        parts = spec.split(":")
        if len(parts) < 7:
            raise SystemExit(f"--dataset needs >=7 fields, got: {spec}")
        label, z_path, pass_cs, cs, prot, weights, config = parts[:7]
        class_dir = parts[7] if len(parts) > 7 else None
        protein_idx = [int(x) for x in prot.split(",")]
        res.append(analyse(label, z_path, pass_cs, cs, protein_idx, weights,
                           config, class_dir, args.n_dummies, args.rep,
                           args.traj, args.downsample, args.apix, args.out,
                           args.run, args.compare_only, args.seed))

    plot_cc(res, args.out)
    with open(os.path.join(args.out, "decode_states_metrics.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\n[done] decode-states -> {args.out}")


if __name__ == "__main__":
    main()
