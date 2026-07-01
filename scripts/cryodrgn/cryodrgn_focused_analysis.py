#!/usr/bin/env python
"""Analyse focused (global K=2) hetero-refinement of the merged CryoSPARC pairs.

The user re-ran CryoSPARC hetero-refinement (K=2, NO focus mask) on the
P6+P10 and P8+P9 particle subsets and uploaded the two output class volumes per
pair.  The decisive question:

    Given ONLY the images of a "merged" pair, does CryoSPARC reproducibly
    reconstruct TWO structurally distinct maps?  And does that difference land
    where cryoDRGN-LDA predicted it would?

This is the orthogonal, image-space validation of the LDA substate result
(which was supervised on the CryoSPARC labels and so could not, by itself,
prove the difference was independent of CryoSPARC's own signal).

Per pair we compute, entirely within the CryoSPARC frame (no cross-method
assumptions):
  * masked real-space CC(class0, class1)         -- low CC => a real split;
  * FSC(class0, class1) with 0.5 / 0.143 crossings;
  * the heteroref difference map (class1 - class0).
Then, as a bonus cross-check (handedness-robust), we resample the cryoDRGN-LDA
difference map to the same box and correlate |heteroref diff| vs |LDA diff|
over the union of their salient regions, trying axis flips to absorb any
handedness/origin convention mismatch.

Run from repo root with cryodrgn-py310 python::

    python scripts/cryodrgn/cryodrgn_focused_analysis.py \
      --pair P6_P10:results_cryodrgn/focused_classification/J1497/P6_P10 \
      --pair P8_P9:results_cryodrgn/focused_classification/J1497/P8_P9 \
      --apix 2.075 \
      -o results_cryodrgn/focused_classification/J1497
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import affine_transform, zoom
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for p in (_REPO, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cryodrgn_decode_states import cross_correlation, read_mrc
from cryodrgn_lda_states import fsc_curve, normalise_in_mask


# --------------------------------------------------------------------------- #
def find_class_volumes(pair_dir):
    c0 = sorted(glob.glob(os.path.join(pair_dir, "*class_00*volume*.mrc")))
    c1 = sorted(glob.glob(os.path.join(pair_dir, "*class_01*volume*.mrc")))
    return (c0[0] if c0 else None), (c1[0] if c1 else None)


def find_lda_diff(pair_dir):
    hits = sorted(glob.glob(os.path.join(pair_dir, "*diff.mrc")))
    return hits[0] if hits else None


def resample_to(vol, n):
    if vol.shape[0] == n:
        return vol
    return zoom(vol, n / vol.shape[0], order=1)


def _apply_rigid(vol, R, shift, center):
    """Resample `vol` by rotation R (about center) then translation `shift`."""
    offset = center - R @ (center + shift)
    return affine_transform(vol, R, offset=offset, order=1, mode="nearest")


def _mcorr(a, b, m):
    a = a[m]; b = b[m]
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else -1.0


def align_rigid(ref, mov, n_search=48, n_rot=500, seed=0):
    """Rigid-body align `mov` onto `ref` (rotation+translation) maximising masked
    CC.  Coarse global rotation search at low resolution, then local Powell
    refinement, then apply the transform at full resolution.

    Returns (aligned_mov_fullres, info_dict).
    """
    n_full = ref.shape[0]
    f = n_search / n_full
    r_s = zoom(ref, f, order=1)
    m_s = zoom(mov, f, order=1)
    c = np.array(r_s.shape, float) / 2.0
    thr = r_s.mean() + 0.5 * r_s.std()
    mask = r_s > thr
    if mask.sum() < 50:
        mask = r_s > r_s.mean()

    raw = _mcorr(m_s, r_s, mask)
    # coarse global rotation search (deterministic set + identity)
    rots = [np.eye(3)] + [R.as_matrix() for R in
                          Rotation.random(n_rot, random_state=seed)]
    best_cc, best_R = -2.0, np.eye(3)
    for R in rots:
        cc = _mcorr(_apply_rigid(m_s, R, np.zeros(3), c), r_s, mask)
        if cc > best_cc:
            best_cc, best_R = cc, R

    # local refinement over (rotvec 3, shift 3) in low-res voxels
    rv0 = Rotation.from_matrix(best_R).as_rotvec()
    x0 = np.concatenate([rv0, np.zeros(3)])

    def neg_cc(x):
        R = Rotation.from_rotvec(x[:3]).as_matrix()
        return -_mcorr(_apply_rigid(m_s, R, x[3:], c), r_s, mask)

    res = minimize(neg_cc, x0, method="Powell",
                   options={"maxiter": 200, "xtol": 1e-3, "ftol": 1e-4})
    R_fin = Rotation.from_rotvec(res.x[:3]).as_matrix()
    shift_full = res.x[3:] / f
    aligned = _apply_rigid(mov, R_fin, shift_full, np.array(mov.shape, float) / 2.0)
    info = {"cc_raw_lowres": raw, "cc_aligned_lowres": float(-res.fun),
            "rot_deg": float(np.degrees(np.linalg.norm(res.x[:3]))),
            "shift_vox": [float(s) for s in shift_full]}
    return aligned.astype(np.float32), info


def best_flip_corr(a, b, mask):
    """Max |corr| of a vs b over axis flips (absorbs handedness/origin flips)."""
    best_c, best_name, best_abs = 0.0, "identity", -1.0
    flips = {
        "identity": b,
        "flipx": b[::-1, :, :],
        "flipy": b[:, ::-1, :],
        "flipz": b[:, :, ::-1],
        "flipxyz": b[::-1, ::-1, ::-1],
    }
    for name, bb in flips.items():
        c = cross_correlation(a, bb, mask)
        if np.isfinite(c) and abs(c) > best_abs:
            best_c, best_name, best_abs = float(c), name, abs(c)
    return best_c, best_name


# --------------------------------------------------------------------------- #
def analyse_pair(tag, pair_dir, apix, out, n_rot=500):
    p0, p1 = find_class_volumes(pair_dir)
    if p0 is None or p1 is None:
        print(f"[{tag}] missing class volumes in {pair_dir}")
        return None
    v0 = read_mrc(p0).astype(np.float32)
    v1 = read_mrc(p1).astype(np.float32)
    n = v0.shape[0]

    # raw (un-aligned) CC: each hetero-refine class is reconstructed in its own
    # pose frame, so this conflates a rigid offset with real structural change.
    ref0 = 0.5 * (v0 + v1)
    mask0 = ref0 > (ref0.mean() + 0.5 * ref0.std())
    cc_raw = cross_correlation(v0, v1, mask0)

    # rigid-body align class1 -> class0, then measure what difference REMAINS
    v1a, ainfo = align_rigid(v0, v1, n_rot=n_rot, seed=0)
    print(f"[{tag}] alignment: raw CC {cc_raw:.3f} -> aligned (lowres) "
          f"{ainfo['cc_aligned_lowres']:.3f} | rot {ainfo['rot_deg']:.1f} deg, "
          f"shift {np.linalg.norm(ainfo['shift_vox']):.1f} vox")
    v1 = v1a

    ref = 0.5 * (v0 + v1)
    mask = ref > (ref.mean() + 0.5 * ref.std())

    cc = cross_correlation(v0, v1, mask)
    freq, fsc, res05 = fsc_curve(v0, v1, apix)
    # 0.143 crossing
    below143 = np.where(fsc < 0.143)[0]
    res143 = float(1.0 / freq[below143[0]]) if below143.size and below143[0] > 0 else np.inf

    diff = normalise_in_mask(v1, mask) - normalise_in_mask(v0, mask)
    diff_rms = float(np.sqrt((diff.ravel()[mask.ravel()] ** 2).mean()))

    # bonus: cross-check vs LDA-predicted difference region
    lda_match, lda_flip, lda_path = np.nan, None, find_lda_diff(pair_dir)
    lda_diff_rs = None
    if lda_path is not None:
        lda_diff = read_mrc(lda_path).astype(np.float32)
        lda_diff_rs = resample_to(lda_diff, n)
        # salient union mask from both |diff| maps
        a = np.abs(diff)
        b = np.abs(lda_diff_rs)
        sal = (a >= np.percentile(a, 99.0)) | (b >= np.percentile(b, 99.0))
        lda_match, lda_flip = best_flip_corr(a, b, sal)

    print(f"[{tag}] aligned class0 vs class1: CC {cc:.3f} | "
          f"FSC0.5 {res05:.1f} A | FSC0.143 {res143:.1f} A | diff RMS {diff_rms:.3f}"
          + ("" if np.isnan(lda_match) else
             f" | |diff| vs LDA-diff corr {lda_match:.3f} ({lda_flip})"))

    return {"pair": tag, "box": int(n), "apix": apix,
            "cc_raw_unaligned": float(cc_raw),
            "align_rot_deg": ainfo["rot_deg"],
            "align_shift_vox": float(np.linalg.norm(ainfo["shift_vox"])),
            "cc_class0_class1": cc, "fsc05_A": res05, "fsc0143_A": res143,
            "diff_rms": diff_rms,
            "lda_region_match": (None if np.isnan(lda_match) else lda_match),
            "lda_region_flip": lda_flip,
            "_v0": v0, "_v1": v1, "_diff": diff, "_mask": mask,
            "_freq": freq, "_fsc": fsc, "_lda_diff": lda_diff_rs}


# --------------------------------------------------------------------------- #
def plot(results, out):
    ok = [r for r in results if r is not None]
    if not ok:
        return
    npairs = len(ok)
    fig, axes = plt.subplots(npairs, 5, figsize=(19, 3.9 * npairs), squeeze=False)
    for row, r in enumerate(ok):
        n = r["box"]
        mid = n // 2
        panels = [(r["_v0"], "class 0", "gray"),
                  (r["_v1"], "class 1", "gray"),
                  (r["_diff"], f"heteroref diff\nCC(c0,c1)={r['cc_class0_class1']:.2f}",
                   "bwr")]
        if r["_lda_diff"] is not None:
            panels.append((r["_lda_diff"],
                           f"LDA-predicted diff\nmatch={r['lda_region_match']:.2f}"
                           f" ({r['lda_region_flip']})", "bwr"))
        for col, (vol, ttl, cmap) in enumerate(panels):
            ax = axes[row][col]
            if cmap == "bwr":
                vmax = np.abs(vol).max() or 1.0
                ax.imshow(vol[mid], cmap=cmap, vmin=-vmax, vmax=vmax)
            else:
                ax.imshow(vol[mid], cmap=cmap)
            ax.set_title(f"{r['pair']}  {ttl}", fontsize=9)
            ax.axis("off")
        axf = axes[row][4]
        axf.plot(r["_freq"], r["_fsc"], color="tab:blue")
        axf.axhline(0.5, color="k", ls="--", lw=0.8)
        axf.axhline(0.143, color="gray", ls=":", lw=0.8)
        axf.set_title(f"FSC class0 vs class1\nFSC0.5 {r['fsc05_A']:.1f} A | "
                      f"0.143 {r['fsc0143_A']:.1f} A", fontsize=9)
        axf.set_xlabel("1/A"); axf.set_ylim(-0.1, 1.05)
    fig.suptitle("Focused (global K=2) hetero-refinement of merged pairs: "
                 "did CryoSPARC re-split them from images alone?", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(out, "focused_heteroref_analysis.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[plot] {p}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", action="append", required=True,
                    help="TAG:PAIR_DIR (repeatable)")
    ap.add_argument("--apix", type=float, default=2.075)
    ap.add_argument("--n-rot", type=int, default=500,
                    help="coarse global rotation samples for alignment")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    results = []
    for spec in args.pair:
        tag, pair_dir = spec.split(":", 1)
        results.append(analyse_pair(tag, pair_dir, args.apix, args.out,
                                    n_rot=args.n_rot))

    plot(results, args.out)
    clean = [{k: v for k, v in r.items() if not k.startswith("_")}
             for r in results if r is not None]
    with open(os.path.join(args.out, "focused_heteroref_metrics.json"), "w") as fh:
        json.dump(clean, fh, indent=2)
    print(f"\n[done] focused hetero-refinement analysis -> {args.out}")


if __name__ == "__main__":
    main()
