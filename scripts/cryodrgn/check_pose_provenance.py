#!/usr/bin/env python
"""Audit cryoDRGN pose provenance: GLOBAL-CONSENSUS vs PER-CLASS (heteroref bias).

cryoDRGN never re-aligns particles -- it trusts the input poses.pkl -- so the
whole "are we biased by the CryoSPARC classification?" question reduces to: were
those poses a single GLOBAL consensus alignment, or the per-class heteroref pose
of each particle's winning class?  This tool answers that from the .cs metadata.

It reports, for a CryoSPARC .cs:
  * which pose fields exist:
        alignments3D/pose        -> a single CONSENSUS pose per particle (SAFE)
        alignments3D_multi/pose  -> a PER-CLASS pose (N, K, 3) (heteroref; biased
                                    if you take the winning class)
        alignments_class_0/pose  -> ab-initio single pose (SAFE)
  * if BOTH are present (or --multi is given): how far the consensus pose differs
    from the winning-class pose (geodesic angle) -- i.e. how much bias per-class
    poses would inject.
  * if --poses poses.pkl is given: which source that poses.pkl was actually built
    from (matches cryoDRGN's expmap()+transpose convention), so you can VERIFY a
    run used the global alignment.

Run with the cryoDRGN env (needs cryodrgn.lie_tools) from repo root::

    # verify a run's poses.pkl came from the consensus (J2708 / J4624 style)
    python scripts/cryodrgn/check_pose_provenance.py \\
      --cs   data/eP30W12_J2708/cryosparc_P30_J2708_passthrough_particles_all_classes_alignments3D.cs \\
      --multi data/eP30W12_J2708/cryosparc_P30_J2708_00042_particles_alignments3D_multi.cs \\
      --poses results_cryodrgn/J2708_real/inputs/poses.pkl

    # just classify a .cs (consensus available or per-class only?)
    python scripts/cryodrgn/check_pose_provenance.py --cs <some>.cs
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import torch
from cryodrgn import lie_tools


def load_cs(path):
    return np.load(path)


def axisangle_to_R(v):
    """(N,3) axis-angle -> (N,3,3) exactly as parse_pose_csparc (expmap then .T)."""
    # struct-field views have record-sized strides -> force a contiguous copy
    arr = np.ascontiguousarray(np.asarray(v), dtype=np.float32)
    R = lie_tools.expmap(torch.tensor(arr)).cpu().numpy()
    return np.transpose(R, (0, 2, 1))


def geo_angle_deg(A, B):
    """Per-row geodesic angle (deg) between two stacks of rotation matrices."""
    M = np.einsum("nij,nkj->nik", A, B)          # A @ B^T
    tr = np.trace(M, axis1=1, axis2=2)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def uid_map(uids):
    return {int(u): i for i, u in enumerate(np.asarray(uids).astype(np.uint64).tolist())}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cs", required=True, help="CryoSPARC .cs with pose fields")
    ap.add_argument("--multi", default=None,
                    help="separate *_alignments3D_multi.cs (per-class) to compare")
    ap.add_argument("--poses", default=None,
                    help="poses.pkl to verify (which source was it built from?)")
    ap.add_argument("--sub", type=int, default=20000,
                    help="subsample for the angle stats (0 = all)")
    ap.add_argument("--bias-deg", type=float, default=1.0,
                    help="consensus-vs-perclass angle above which per-class is "
                         "meaningfully biased (default 1 deg)")
    args = ap.parse_args()

    cs = load_cs(args.cs)
    names = cs.dtype.names or ()
    has_single = "alignments3D/pose" in names
    has_multi_in = "alignments3D_multi/pose" in names
    has_abinit = "alignments_class_0/pose" in names
    print(f"[cs] {args.cs}")
    print(f"[cs] {len(cs):,} rows")
    print(f"[cs] alignments3D/pose (consensus): {'YES' if has_single else 'no'}")
    print(f"[cs] alignments3D_multi/pose (per-class): {'YES' if has_multi_in else 'no'}")
    print(f"[cs] alignments_class_0/pose (ab-initio): {'YES' if has_abinit else 'no'}")

    # consensus rotations (from this .cs)
    R_cons = None
    if has_single:
        R_cons = axisangle_to_R(cs["alignments3D/pose"])
    elif has_abinit:
        R_cons = axisangle_to_R(cs["alignments_class_0/pose"])

    # per-class winning rotations (from this .cs or --multi), aligned to cs order
    R_win = None
    class_spread = None
    mm = cs if has_multi_in else (load_cs(args.multi) if args.multi else None)
    if mm is not None and "alignments3D_multi/pose" in (mm.dtype.names or ()):
        pose_mc = np.asarray(mm["alignments3D_multi/pose"], dtype=np.float32)  # (N,K,3)
        post = np.asarray(mm["alignments3D_multi/class_posterior"], dtype=np.float64)
        best = post.argmax(1)
        win = pose_mc[np.arange(len(pose_mc)), best, :]
        R_win_mm = axisangle_to_R(win)
        # how much do the K per-class poses disagree (as rotations) per particle?
        Rk = [axisangle_to_R(pose_mc[:, k, :]) for k in range(pose_mc.shape[1])]
        # spread = mean geodesic angle of each class pose to the winning pose
        sp = np.mean([geo_angle_deg(Rk[k], R_win_mm) for k in range(len(Rk))], axis=0)
        class_spread = sp
        # reorder mm rows to cs order by uid if separate file
        if mm is cs:
            R_win = R_win_mm
        else:
            m = uid_map(mm["uid"])
            order = np.array([m.get(int(u), -1) for u in
                              np.asarray(cs["uid"]).astype(np.uint64).tolist()])
            ok = order >= 0
            R_win = np.full_like(R_cons if R_cons is not None else R_win_mm, np.nan)
            R_win[ok] = R_win_mm[order[ok]]
            print(f"[multi] matched {ok.sum():,}/{len(cs):,} cs rows to --multi by uid")

    # subsample indices for angle stats
    N = len(cs)
    rng = np.random.default_rng(0)
    idx = (rng.choice(N, args.sub, replace=False)
           if args.sub and N > args.sub else np.arange(N))

    # consensus vs per-class divergence
    if R_cons is not None and R_win is not None:
        a = geo_angle_deg(R_cons[idx], R_win[idx])
        a = a[~np.isnan(a)]
        print("\n[bias] consensus-pose vs winning-class-pose divergence (deg):")
        print(f"       median {np.median(a):.2f}  mean {a.mean():.2f}  "
              f"p90 {np.percentile(a, 90):.2f}  max {a.max():.2f}")
        print(f"       fraction > {args.bias_deg} deg: {(a > args.bias_deg).mean()*100:.1f}%")
        if np.median(a) > args.bias_deg:
            print(f"       => per-class poses DIFFER from consensus: using them WOULD bias.")
        else:
            print(f"       => per-class ~= consensus here (small bias).")
    if class_spread is not None:
        cs_sub = class_spread[idx if len(class_spread) == N else np.arange(len(class_spread))]
        print(f"[bias] within-particle spread of the {mm['alignments3D_multi/pose'].shape[1]} "
              f"class poses: median {np.median(cs_sub):.2f} deg")

    # verify a poses.pkl
    if args.poses:
        with open(args.poses, "rb") as fh:
            obj = pickle.load(fh)
        R_pkl = np.asarray(obj[0] if isinstance(obj, tuple) else obj, dtype=np.float32)
        print(f"\n[poses] {args.poses}: {R_pkl.shape}")
        if R_pkl.shape[0] != N:
            print(f"        NOTE poses has {R_pkl.shape[0]} rows, cs has {N}; "
                  "comparing the overlap in order (assumes poses parsed from THIS cs).")
        m = min(R_pkl.shape[0], N)
        sidx = idx[idx < m]
        verdicts = []
        if R_cons is not None:
            ac = geo_angle_deg(R_pkl[sidx], R_cons[sidx]); ac = ac[~np.isnan(ac)]
            print(f"[poses] median angle to CONSENSUS pose: {np.median(ac):.3f} deg")
            verdicts.append(("consensus", np.median(ac)))
        if R_win is not None:
            aw = geo_angle_deg(R_pkl[sidx], R_win[sidx]); aw = aw[~np.isnan(aw)]
            print(f"[poses] median angle to WINNING-CLASS pose: {np.median(aw):.3f} deg")
            verdicts.append(("per-class", np.median(aw)))
        if verdicts:
            src, ang = min(verdicts, key=lambda t: t[1])
            print("\n================ VERDICT ================")
            if ang < 0.5:
                if src == "consensus":
                    print("poses.pkl was built from the GLOBAL CONSENSUS alignment. SAFE.")
                else:
                    print("poses.pkl was built from PER-CLASS (winning-class) poses. "
                          "BIASED -> rebuild from a C1 consensus refinement.")
            else:
                print(f"poses.pkl matches neither source well (min {ang:.2f} deg); "
                      "check uid order / --D scaling / correct .cs.")
            print("=========================================")
    else:
        print("\n================ VERDICT ================")
        if has_single or has_abinit:
            print("This .cs HAS a single consensus pose (alignments3D/pose). SAFE to use.")
            if R_win is not None and R_cons is not None:
                print("(pass --poses to confirm the run's poses.pkl used it, not the multi.)")
        elif has_multi_in or args.multi:
            print("This .cs has ONLY per-class poses (alignments3D_multi). "
                  "NEEDS a C1 consensus refinement to get an unbiased pose.")
        else:
            print("No recognized pose fields found in this .cs.")
        print("=========================================")


if __name__ == "__main__":
    main()
