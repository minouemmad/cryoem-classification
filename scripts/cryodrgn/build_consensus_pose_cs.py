#!/usr/bin/env python
"""Build a single-class CryoSPARC .cs (consensus poses) from a hetero-refinement
``alignments3D_multi`` file, so that ``cryodrgn parse_pose_csparc`` can be run on
it unchanged.

CryoSPARC hetero-refinement stores a *per-class* pose for every particle
(``alignments3D_multi/pose`` has shape ``(N, num_classes, 3)``); cryoDRGN's
``parse_pose_csparc`` instead expects a single-class refinement with
``alignments3D/pose`` of shape ``(N, 3)``.  For each particle we take the pose of
its most-probable class (argmax of ``alignments3D_multi/class_posterior``) --
the pose you would actually reconstruct that particle with -- and re-emit it
under the standard ``alignments3D/*`` field names.

This runs on metadata only (no images needed), so it can be done locally; the
resulting ``poses.pkl`` is produced by the subsequent ``parse_pose_csparc`` call
using cryoDRGN's exact rotation/shift conventions.

Run with the cryoDRGN env from the repo root::

    python scripts/cryodrgn/build_consensus_pose_cs.py \
      --multi data/J264/cryosparc_P7_J264_00062_particles_alignments3D_multi.cs \
      -o data/J264/cryosparc_P7_J264_consensus_pose.cs

    # then, still local (metadata only):
    cryodrgn parse_pose_csparc data/J264/cryosparc_P7_J264_consensus_pose.cs \
      -D 320 -o results_cryodrgn/J264_real/inputs/poses.pkl
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--multi", required=True,
                    help="CryoSPARC *_alignments3D_multi.cs (hetero-refine).")
    ap.add_argument("-o", "--out", required=True,
                    help="output single-class .cs (consensus poses).")
    args = ap.parse_args()

    m = np.load(args.multi)
    names = m.dtype.names or ()
    for req in ("alignments3D_multi/pose", "alignments3D_multi/shift",
                "alignments3D_multi/class_posterior"):
        if req not in names:
            raise SystemExit(f"ERROR: {req} not in {args.multi}")

    post = np.asarray(m["alignments3D_multi/class_posterior"], dtype=np.float64)
    pose = np.asarray(m["alignments3D_multi/pose"], dtype=np.float32)   # (N,K,3)
    shift = np.asarray(m["alignments3D_multi/shift"], dtype=np.float32)  # (N,K,2)
    n, k = post.shape
    idx = np.arange(n)
    best = post.argmax(axis=1)                                          # (N,)

    pose_best = pose[idx, best, :]                                      # (N,3)
    shift_best = shift[idx, best, :]                                    # (N,2)
    if "alignments3D_multi/psize_A" in names:
        psize = np.asarray(m["alignments3D_multi/psize_A"], dtype=np.float32)
    else:
        psize = np.full(n, np.nan, dtype=np.float32)
    split = (np.asarray(m["alignments3D_multi/split"])
             if "alignments3D_multi/split" in names
             else np.zeros(n, dtype=np.uint32))

    dt = [("uid", m["uid"].dtype),
          ("alignments3D/split", split.dtype),
          ("alignments3D/shift", "<f4", (2,)),
          ("alignments3D/pose", "<f4", (3,)),
          ("alignments3D/psize_A", "<f4")]
    out = np.empty(n, dtype=np.dtype(dt))
    out["uid"] = m["uid"]
    out["alignments3D/split"] = split
    out["alignments3D/shift"] = shift_best
    out["alignments3D/pose"] = pose_best
    out["alignments3D/psize_A"] = psize

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as fh:
        np.save(fh, out)

    frac = np.bincount(best, minlength=k) / n
    print(f"[build] {args.out}  ({n:,} particles)")
    print(f"[build] assigned-class fractions (argmax posterior): "
          f"{np.round(frac, 3).tolist()}")
    print(f"[build] pose range {pose_best.min():.3f}..{pose_best.max():.3f} "
          f"(axis-angle); shift range {shift_best.min():.2f}.."
          f"{shift_best.max():.2f}")


if __name__ == "__main__":
    main()
