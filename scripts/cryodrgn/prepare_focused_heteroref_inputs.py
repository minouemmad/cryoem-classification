#!/usr/bin/env python
"""Prepare CryoSPARC focused-classification inputs from LDA endpoint maps.

Given a cryoDRGN/CryoSPARC aligned dataset and an existing LDA decode directory
(e.g. results_cryodrgn/lda_states/J1497), this script creates:

1) CryoSPARC-importable particle subsets (.cs) for each requested class pair:
   - pair_Pa_Pb_particles.cs   (combined pair; use this as hetero-refine input, K=2)
   - pair_Pa_particles.cs      (class-a seed subset)
   - pair_Pb_particles.cs      (class-b seed subset)

2) Focus masks from LDA endpoint difference maps (hi - lo):
   - pair_Pa_Pb_diff.mrc
   - pair_Pa_Pb_mask_hard.mrc   (largest connected |diff| region)
   - pair_Pa_Pb_mask_soft.mrc   (gaussian-softened mask in [0,1])
   - pair_Pa_Pb_mask_pos_soft.mrc / _neg_soft.mrc (optional sign-specific masks)

These are intended for the immediate CryoSPARC follow-up:
  import pair particles -> focused hetero-refine K=2 with the soft mask.

Run from repo root with cryodrgn-py310 python::

    python scripts/cryodrgn/prepare_focused_heteroref_inputs.py \
      --dataset "J1497:results_cryodrgn/J1497_real/train/z.100.pkl:data/gP25W6J1497_passthrough_particles_all_classes.cs:data/cryosparc_P25_J1497_00000_particles.cs:6,7,8,9,10:data/J1497_classes" \
      --lda-dir results_cryodrgn/lda_states/J1497 \
      --lda-prefix J1497_lda \
      --pairs 6-10,8-9 \
      --n-dummies 6 \
      --mask-quantile 99.5 --dilate 2 --close 1 --soft-sigma 1.5 \
      --apix 4.15 \
      -o results_cryodrgn/focused_classification/J1497
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, gaussian_filter, label

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for p in (_REPO, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from cryodrgn.mrcfile import write_mrc

import cryodrgn_latent_gmm as clg
from cryodrgn_decode_states import load_official_membership, read_mrc


def save_cs(path: str, arr: np.ndarray) -> None:
    with open(path, "wb") as fh:
        np.save(fh, arr)


def parse_pairs(spec: str):
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        a, b = tok.split("-")
        out.append((int(a), int(b)))
    return out


def read_lda_labels(path: str):
    d = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            i, name = line.split("\t", 1)
            d[name] = int(i)
    return d


def largest_component(mask: np.ndarray) -> np.ndarray:
    cc, n = label(mask)
    if n == 0:
        return mask
    counts = np.bincount(cc.ravel())
    counts[0] = 0
    best = counts.argmax()
    return cc == best


def make_masks(diff: np.ndarray, q: float, close_iter: int, dilate_iter: int,
               soft_sigma: float):
    ad = np.abs(diff)
    thr = np.percentile(ad, q)
    hard = ad >= thr
    if close_iter > 0:
        hard = binary_closing(hard, iterations=close_iter)
    if dilate_iter > 0:
        hard = binary_dilation(hard, iterations=dilate_iter)
    hard = largest_component(hard)

    soft = gaussian_filter(hard.astype(np.float32), soft_sigma)
    if soft.max() > 0:
        soft = soft / soft.max()

    pos = (diff > 0) & hard
    neg = (diff < 0) & hard
    pos_soft = gaussian_filter(pos.astype(np.float32), soft_sigma)
    neg_soft = gaussian_filter(neg.astype(np.float32), soft_sigma)
    if pos_soft.max() > 0:
        pos_soft = pos_soft / pos_soft.max()
    if neg_soft.max() > 0:
        neg_soft = neg_soft / neg_soft.max()
    return hard.astype(np.float32), soft.astype(np.float32), pos_soft.astype(np.float32), neg_soft.astype(np.float32), float(thr)


def find_volume(lda_dir: str, lda_prefix: str, idx0: int):
    # volumes are 1-based in filenames
    hits = sorted(glob.glob(os.path.join(lda_dir, f"{lda_prefix}{idx0 + 1:03d}.mrc")))
    hits += sorted(glob.glob(os.path.join(lda_dir, f"{lda_prefix}{idx0 + 1}.mrc")))
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    help="LABEL:Z_PKL:PASSTHROUGH_CS:CS:PROT_IDX[:CLASS_DIR]")
    ap.add_argument("--lda-dir", required=True,
                    help="Directory with LDA decoded maps + *_lda_labels.txt")
    ap.add_argument("--lda-prefix", required=True,
                    help="Volume prefix used by LDA decode, e.g. J1497_lda")
    ap.add_argument("--pairs", default="6-10,8-9")
    ap.add_argument("--n-dummies", type=int, default=6)
    ap.add_argument("--mask-quantile", type=float, default=99.5)
    ap.add_argument("--close", type=int, default=1)
    ap.add_argument("--dilate", type=int, default=2)
    ap.add_argument("--soft-sigma", type=float, default=1.5)
    ap.add_argument("--apix", type=float, default=4.15)
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    parts = args.dataset.split(":")
    if len(parts) < 5:
        raise SystemExit("--dataset must be LABEL:Z:PASSTHROUGH_CS:CS:PROT_IDX[:CLASS_DIR]")
    label, z_path, pass_cs_path, cs_path, prot = parts[:5]
    class_dir = parts[5] if len(parts) > 5 else None
    protein_idx = [int(x) for x in prot.split(",")]
    pairs = parse_pairs(args.pairs)

    # Align latent rows to protein class posteriors and map to passthrough rows by uid.
    z = clg.load_latent(z_path)
    z_a, cryo_post, cryo_hard, uid_a, n_prot = clg.align_z_to_posteriors(
        z, pass_cs_path, cs_path, args.n_dummies, protein_idx)

    member = load_official_membership(class_dir, protein_idx)
    if member is not None:
        official = np.array([member.get(int(u), -1) for u in uid_a.tolist()])
        ok = official >= 0
        if ok.any():
            agree = float(np.mean(official[ok] == cryo_hard[ok]))
            print(f"[member] matched {ok.sum():,}/{len(official):,}; agreement {agree*100:.1f}%")
            cryo_hard = np.where(ok, official, cryo_hard)

    pass_cs = np.load(pass_cs_path)
    pass_uid = pass_cs["uid"].astype(np.uint64)
    uid_to_row = {int(u): i for i, u in enumerate(pass_uid.tolist())}

    lda_labels_path = os.path.join(args.lda_dir, f"{label}_lda_labels.txt")
    if not os.path.exists(lda_labels_path):
        # fallback: first *_lda_labels.txt
        cands = sorted(glob.glob(os.path.join(args.lda_dir, "*_lda_labels.txt")))
        if not cands:
            raise SystemExit(f"No *_lda_labels.txt found in {args.lda_dir}")
        lda_labels_path = cands[0]
    lmap = read_lda_labels(lda_labels_path)

    idx_of = {p: j for j, p in enumerate(protein_idx)}
    summary = []

    for pa, pb in pairs:
        if pa not in idx_of or pb not in idx_of:
            print(f"[pair] skip P{pa}-P{pb}: not in protein_idx")
            continue
        ja, jb = idx_of[pa], idx_of[pb]
        tag = f"P{pa}_P{pb}"
        pair_dir = os.path.join(args.out, tag)
        os.makedirs(pair_dir, exist_ok=True)

        # --- export pair subsets ------------------------------------------------
        ma = cryo_hard == ja
        mb = cryo_hard == jb
        mp = ma | mb

        def rows_for(mask):
            uu = uid_a[mask].astype(np.uint64)
            rows = [uid_to_row.get(int(u), -1) for u in uu.tolist()]
            rows = np.array([r for r in rows if r >= 0], dtype=np.intp)
            return rows

        rows_a = rows_for(ma)
        rows_b = rows_for(mb)
        rows_p = rows_for(mp)

        save_cs(os.path.join(pair_dir, f"pair_P{pa}_particles.cs"), pass_cs[rows_a])
        save_cs(os.path.join(pair_dir, f"pair_P{pb}_particles.cs"), pass_cs[rows_b])
        save_cs(os.path.join(pair_dir, f"pair_P{pa}_P{pb}_particles.cs"), pass_cs[rows_p])

        # --- build focused masks from LDA endpoint volumes ----------------------
        key_lo = f"{pa}_{pb}_lo"
        key_hi = f"{pa}_{pb}_hi"
        if key_lo not in lmap or key_hi not in lmap:
            # allow reversed pair in labels
            key_lo = f"{pb}_{pa}_lo"
            key_hi = f"{pb}_{pa}_hi"
            if key_lo not in lmap or key_hi not in lmap:
                print(f"[pair] skip mask for P{pa}-P{pb}: no lo/hi labels in {lda_labels_path}")
                summary.append({
                    "pair": f"P{pa}-P{pb}",
                    "n_pair": int(len(rows_p)),
                    "n_a": int(len(rows_a)),
                    "n_b": int(len(rows_b)),
                    "mask_written": False,
                })
                continue

        vlo = find_volume(args.lda_dir, args.lda_prefix, lmap[key_lo])
        vhi = find_volume(args.lda_dir, args.lda_prefix, lmap[key_hi])
        if vlo is None or vhi is None:
            print(f"[pair] skip mask for P{pa}-P{pb}: endpoint maps missing")
            summary.append({
                "pair": f"P{pa}-P{pb}",
                "n_pair": int(len(rows_p)),
                "n_a": int(len(rows_a)),
                "n_b": int(len(rows_b)),
                "mask_written": False,
            })
            continue

        lo = read_mrc(vlo)
        hi = read_mrc(vhi)
        diff = hi - lo
        hard, soft, pos_soft, neg_soft, thr = make_masks(
            diff, args.mask_quantile, args.close, args.dilate, args.soft_sigma)

        write_mrc(os.path.join(pair_dir, f"pair_P{pa}_P{pb}_diff.mrc"),
                  diff.astype(np.float32), is_vol=True, Apix=args.apix)
        write_mrc(os.path.join(pair_dir, f"pair_P{pa}_P{pb}_mask_hard.mrc"),
                  hard, is_vol=True, Apix=args.apix)
        write_mrc(os.path.join(pair_dir, f"pair_P{pa}_P{pb}_mask_soft.mrc"),
                  soft, is_vol=True, Apix=args.apix)
        write_mrc(os.path.join(pair_dir, f"pair_P{pa}_P{pb}_mask_pos_soft.mrc"),
                  pos_soft, is_vol=True, Apix=args.apix)
        write_mrc(os.path.join(pair_dir, f"pair_P{pa}_P{pb}_mask_neg_soft.mrc"),
                  neg_soft, is_vol=True, Apix=args.apix)

        print(f"[P{pa}-P{pb}] exported particles A/B/pair = "
              f"{len(rows_a):,}/{len(rows_b):,}/{len(rows_p):,} | "
              f"mask voxels {int(hard.sum()):,} (|diff| >= {thr:.4g})")
        summary.append({
            "pair": f"P{pa}-P{pb}",
            "n_pair": int(len(rows_p)),
            "n_a": int(len(rows_a)),
            "n_b": int(len(rows_b)),
            "diff_threshold_abs": thr,
            "hard_mask_voxels": int(hard.sum()),
            "mask_written": True,
            "pair_dir": pair_dir,
        })

    with open(os.path.join(args.out, f"{label}_focused_inputs.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n[done] focused hetero-refine inputs -> {args.out}")


if __name__ == "__main__":
    main()
