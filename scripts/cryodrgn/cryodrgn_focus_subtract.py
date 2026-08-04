#!/usr/bin/env python
"""Pose-aware FOCUSED SIGNAL SUBTRACTION inside cryoDRGN's own forward model.

Goal: make cryoDRGN more sensitive to a subtle moving region by removing the
dominant *rigid* density that pins the latent to a few big blobs.  For every
particle we subtract the CTF-modulated projection of the density OUTSIDE the
focus mask -- projected through *that particle's own pose* -- leaving images
dominated by the moving region.  Train cryoDRGN on the residual stack (same
poses.pkl / ctf.pkl); cluster the sharper latent; the cluster labels apply to
the ORIGINAL particles, so "recombination" is automatic (refine the original
particles per cluster).

Everything is done in cryoDRGN's exact conventions -- it reuses
``ctf.load_ctf_for_training`` (which rescales Apix to the box), ``compute_ctf``,
the ``lattice.coords @ rot`` central-slice mapping, and ``translate_ht`` -- so
the residual is consistent with how train_vae will read it.  A per-particle
signed least-squares scale absorbs the sign/normalisation gap between the decoded
consensus volume and the raw images, so no invert/normalise bookkeeping is needed.

NO CryoSPARC.  NO change to cryoDRGN source (it imports the installed package).

Validate the geometry first (works anywhere, no data needed)::

    python scripts/cryodrgn/cryodrgn_focus_subtract.py --self-test

Then run it (on hudson, cryodrgn env)::

    python scripts/cryodrgn/cryodrgn_focus_subtract.py \\
      --particles results_cryodrgn/J1442_gP25_WT_POSE_BIAS/inputs/particles.128.mrcs \\
      --poses     results_cryodrgn/J1442_gP25_WT_POSE_BIAS/inputs/poses.pkl \\
      --ctf       results_cryodrgn/J1442_gP25_WT_POSE_BIAS/inputs/ctf.pkl \\
      --consensus results_cryodrgn/J1442_gP25_WT_POSE_BIAS/focus/consensus.mrc \\
      --mask      results_cryodrgn/J1442_gP25_WT_POSE_BIAS/focus/focus_mask.mrc \\
      -o results_cryodrgn/J1442_gP25_WT_POSE_BIAS/inputs/particles.128.focus.mrcs
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from cryodrgn import ctf as ctf_mod
from cryodrgn.fft import ht2_center, symmetrize_ht, iht2_center, htn_center
from cryodrgn.lattice import Lattice
from cryodrgn.mrcfile import parse_mrc, write_mrc, MRCHeader
from cryodrgn.source import ImageSource
from cryodrgn.utils import load_pkl


# --------------------------------------------------------------------------- #
# 3D Hartley of a volume, symmetrised to the odd (D+1) lattice
# --------------------------------------------------------------------------- #
def build_ht3(vol: torch.Tensor) -> torch.Tensor:
    """(D,D,D) real volume -> (1,1,D+1,D+1,D+1) symmetrised Hartley for grid_sample.

    Symmetrisation copies the first plane/row/col to a new last index so the DC
    term sits at the exact centre of an ODD grid (index D/2), matching
    grid_sample(align_corners=True) where the normalised coord 0 maps to centre.
    """
    ht = htn_center(vol)                       # (D,D,D), DC at index D//2
    D = ht.shape[-1]
    s = torch.empty((D + 1, D + 1, D + 1), dtype=ht.dtype)
    s[:D, :D, :D] = ht
    s[-1, :, :] = s[0, :, :]
    s[:, -1, :] = s[:, 0, :]
    s[:, :, -1] = s[:, :, 0]
    return s[None, None]                        # (1,1,D+1,D+1,D+1)


def extract_slices(ht3: torch.Tensor, coords: torch.Tensor,
                   rot: torch.Tensor) -> torch.Tensor:
    """Central-slice a symmetrised 3D Hartley at rotated lattice coords.

    ht3:    (1,1,Ds,Ds,Ds)  symmetrised Hartley (Ds = D+1, odd)
    coords: (N,3) lattice coords in [-0.5, 0.5]  (extent=0.5, z-plane)
    rot:    (B,3,3) cryoDRGN pose rotations   ->  slice at (coords @ rot)
    returns (B,N) Hartley values of the projection
    """
    B = rot.shape[0]
    N = coords.shape[0]
    cc = torch.matmul(coords.unsqueeze(0).expand(B, -1, -1), rot)   # (B,N,3)
    # coords in [-0.5,0.5]; grid_sample wants [-1,1]  ->  *2  (extent=0.5)
    g = 2.0 * cc
    # grid_sample last-dim order is (x,y,z) = (W,H,D); volume axes are (z,y,x)
    grid = torch.stack([g[..., 0], g[..., 1], g[..., 2]], dim=-1)   # (B,N,3)
    grid = grid.view(B, 1, 1, N, 3)
    out = F.grid_sample(ht3.expand(B, -1, -1, -1, -1), grid,
                        mode="bilinear", align_corners=True,
                        padding_mode="zeros")                       # (B,1,1,1,N)
    return out.view(B, N)


def compute_ctf_batch(freqs2d: torch.Tensor, ctf_params: torch.Tensor) -> torch.Tensor:
    """freqs2d (N,2) cycles/pixel ; ctf_params (B,8)=[Apix,dfu,dfv,dfang,kV,cs,w,ph]."""
    B = ctf_params.shape[0]
    N = freqs2d.shape[0]
    apix = ctf_params[:, 0].view(B, 1, 1)
    freqs = freqs2d.unsqueeze(0).expand(B, N, 2) / apix            # (B,N,2)
    c = ctf_mod.compute_ctf(freqs, *torch.split(ctf_params[:, 1:], 1, 1))
    return c.view(B, N)


def subtract_batch(x, rot, trans, ctf_params, ht3, L, coords, fitmask, device):
    """Return residual real-space images (B,D,D) after focused subtraction."""
    B, D, _ = x.shape
    Ds = D + 1
    y = symmetrize_ht(ht2_center(x)).view(B, -1)                   # (B, Ds*Ds)
    slc = extract_slices(ht3, coords, rot)                        # (B, Ds*Ds)
    c = compute_ctf_batch(L.freqs2d, ctf_params)                 # (B, Ds*Ds)
    proj = c * slc                                                # centred-volume projection

    t = trans.unsqueeze(1)                                        # (B,1,2)
    yc = L.translate_ht(y, t).view(B, -1)                        # centre the image
    # signed per-particle least-squares scale over the protein disk
    m = fitmask
    num = (yc[:, m] * proj[:, m]).sum(1)
    den = (proj[:, m] * proj[:, m]).sum(1) + 1e-8
    alpha = (num / den).view(B, 1)
    proj_raw = L.translate_ht((alpha * proj), (-trans).unsqueeze(1)).view(B, -1)
    resid = (y - proj_raw).view(B, Ds, Ds)
    resid = resid[:, :D, :D].contiguous()                        # de-symmetrise
    return iht2_center(resid), alpha.view(B)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(args):
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[dev] {device}")

    # --- volumes -> outside-mask density -> 3D Hartley -------------------- #
    V0, _ = parse_mrc(args.consensus)
    V0 = np.asarray(V0, dtype=np.float32)
    D = V0.shape[-1]
    M, _ = parse_mrc(args.mask)
    M = np.asarray(M, dtype=np.float32)
    if M.shape != V0.shape:
        raise SystemExit(f"mask {M.shape} != consensus {V0.shape}")
    M = np.clip(M, 0.0, 1.0)
    if args.invert_mask:                      # subtract the mask region instead
        V_out = V0 * M
    else:                                     # default: subtract OUTSIDE the mask
        V_out = V0 * (1.0 - M)
    ht3 = build_ht3(torch.from_numpy(V_out)).to(device)
    print(f"[vol] box {D}, subtracting {'inside' if args.invert_mask else 'outside'} "
          f"mask density (mask covers {(M>0.5).mean()*100:.1f}% of box)")

    # --- lattice / poses / ctf ------------------------------------------- #
    L = Lattice(D + 1, extent=0.5, device=device)
    coords = L.coords                                            # (Ds^2,3) in [-0.5,0.5]
    fitmask = L.get_circular_mask(D // 2).to(device)            # protein disk for scale fit

    poses = load_pkl(args.poses)
    if isinstance(poses, tuple):
        rots_np = np.asarray(poses[0], dtype=np.float32)
        trans_np = np.asarray(poses[1], dtype=np.float32) if len(poses) > 1 else None
    else:
        rots_np, trans_np = np.asarray(poses, dtype=np.float32), None
    src = ImageSource.from_file(args.particles, lazy=True, datadir=args.datadir)
    N = src.n
    if trans_np is None:
        trans_np = np.zeros((N, 2), dtype=np.float32)
    if len(rots_np) != N or len(trans_np) != N:
        raise SystemExit(f"poses ({len(rots_np)}) != particles ({N})")

    ctf_params = ctf_mod.load_ctf_for_training(D, args.ctf)     # (N,8), Apix rescaled
    ctf_params = np.asarray(ctf_params, dtype=np.float32)
    if len(ctf_params) != N:
        raise SystemExit(f"ctf ({len(ctf_params)}) != particles ({N})")

    idx = np.arange(N)
    if args.first:
        idx = idx[: args.first]
        N = len(idx)
        print(f"[lim] processing first {N} particles only")

    # --- output MRC (streamed) ------------------------------------------- #
    header = MRCHeader.make_default_header(nz=N, ny=D, nx=D, dtype=np.float32,
                                           is_vol=False, Apix=float(ctf_params[0, 0]))
    f = open(args.o, "wb")
    header.write(f)                            # 1024-byte header (+ ext header)
    alphas = []

    bs = args.batch_size
    for i in range(0, N, bs):
        j = min(i + bs, N)
        bidx = idx[i:j]
        x = src.images(bidx).to(device).float()                # (B,D,D)
        rot = torch.from_numpy(rots_np[bidx]).to(device)
        trans = torch.from_numpy(trans_np[bidx]).to(device)
        cp = torch.from_numpy(ctf_params[bidx]).to(device)
        with torch.no_grad():
            resid, alpha = subtract_batch(x, rot, trans, cp, ht3, L, coords, fitmask, device)
        f.write(resid.cpu().numpy().astype(np.float32).tobytes())
        alphas.append(alpha.cpu().numpy())
        if (i // bs) % 20 == 0:
            print(f"  {j}/{N}  ({j/N*100:.0f}%)  mean|scale|={np.abs(alpha.cpu().numpy()).mean():.3g}")
    f.close()
    alphas = np.concatenate(alphas)
    print(f"[out] {args.o}  ({N} residual particles, box {D})")
    print(f"[fit] per-particle scale: mean {alphas.mean():.3g}  "
          f"sd {alphas.std():.3g}  (sign absorbs invert/normalisation)")
    print("Next: train_vae on this residual stack with the SAME poses.pkl/ctf.pkl "
          "(point PARTICLES at it), then cluster + score as usual.")


# --------------------------------------------------------------------------- #
# Self-test: synthetic volume -> simulate particles -> subtract -> residual
# --------------------------------------------------------------------------- #
def self_test(D=64, B=32, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device("cpu")

    # synthetic volume: big rigid "core" + small "mobile" blob
    ax = torch.arange(D) - D // 2
    zz, yy, xx = torch.meshgrid(ax, ax, ax, indexing="ij")

    def blob(cz, cy, cx, s, a):
        return a * torch.exp(-((zz-cz)**2 + (yy-cy)**2 + (xx-cx)**2) / (2*s*s))
    core = blob(0, 0, 0, 9, 1.0) + blob(0, 8, -6, 6, 0.8)
    mobile = blob(-2, -12, 10, 4, 0.7)
    V = (core + mobile).float()
    Mmask = (blob(-2, -12, 10, 6, 1.0) > 0.15).float()          # mask around mobile

    L = Lattice(D + 1, extent=0.5, device=dev)
    coords = L.coords
    fitmask = L.get_circular_mask(D // 2)

    # random poses (proper rotations via QR), small shifts
    A = torch.randn(B, 3, 3)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(R, dim1=1, dim2=2)).unsqueeze(1)
    dets = torch.linalg.det(Q)
    Q[dets < 0, :, 0] *= -1
    rot = Q
    trans = (torch.rand(B, 2) - 0.5) * 4.0

    # synthetic CTF params (Apix, dfu,dfv,dfang,kV,cs,w,ph)
    apix = 3.0
    cp = torch.tensor([[apix, 12000., 12000., 0., 300., 2.7, 0.1, 0.]]).repeat(B, 1)

    # simulate a particle as CTF*projection(full V), centred then un-centred
    ht3_full = build_ht3(V)
    slc = extract_slices(ht3_full, coords, rot)
    c = compute_ctf_batch(L.freqs2d, cp)
    proj = c * slc
    y_raw = L.translate_ht(proj, (-trans).unsqueeze(1)).view(B, (D+1), (D+1))
    x = iht2_center(y_raw[:, :D, :D].contiguous())              # synthetic real particles

    def energy(t):
        return (t.view(B, -1) ** 2).sum(1)

    # (1) subtract the WHOLE volume -> residual should ~vanish
    ht3_all = build_ht3(V)
    r_all, a_all = subtract_batch(x, rot, trans, cp, ht3_all, L, coords, fitmask, dev)
    ratio_all = (energy(r_all) / energy(x)).mean().item()

    # (2) subtract only OUTSIDE the mask -> residual keeps the mobile blob
    ht3_out = build_ht3(V * (1 - Mmask))
    r_out, a_out = subtract_batch(x, rot, trans, cp, ht3_out, L, coords, fitmask, dev)
    ratio_out = (energy(r_out) / energy(x)).mean().item()

    # reference: projection of ONLY the mobile density (what (2) should resemble)
    ht3_mob = build_ht3(V * Mmask)
    proj_mob = compute_ctf_batch(L.freqs2d, cp) * extract_slices(ht3_mob, coords, rot)
    ymob = L.translate_ht(proj_mob, (-trans).unsqueeze(1)).view(B, (D+1), (D+1))
    xmob = iht2_center(ymob[:, :D, :D].contiguous())
    corr = torch.mean(torch.stack([
        torch.corrcoef(torch.stack([r_out[i].flatten(), xmob[i].flatten()]))[0, 1]
        for i in range(B)])).item()

    print("=== focused-subtraction self-test ===")
    print(f"(1) subtract WHOLE volume  -> residual/particle energy = {ratio_all:.4f}  "
          f"(want << 1, ideally < 0.05)")
    print(f"(2) subtract OUTSIDE mask   -> residual/particle energy = {ratio_out:.4f}  "
          f"(should retain the mobile signal)")
    print(f"    residual (2) vs true mobile-only projection: corr = {corr:.3f}  "
          f"(want close to 1)")
    ok = ratio_all < 0.08 and corr > 0.9
    print("RESULT:", "PASS - geometry/CTF/pose/scale consistent" if ok else
          "CHECK - residual not vanishing; axis-order or convention needs a look")
    return ok


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="run the synthetic geometry validation and exit")
    ap.add_argument("--particles", help="input particle .mrcs (training box)")
    ap.add_argument("--poses", help="poses.pkl (same as training)")
    ap.add_argument("--ctf", help="ctf.pkl (same as training)")
    ap.add_argument("--consensus", help="consensus.mrc from make_focus_mask.py")
    ap.add_argument("--mask", help="focus_mask.mrc (mobile region) from make_focus_mask.py")
    ap.add_argument("--invert-mask", action="store_true",
                    help="subtract INSIDE the mask instead of outside (rarely wanted)")
    ap.add_argument("--datadir", default=None, help="image datadir if particles is .star/.txt")
    ap.add_argument("--first", type=int, default=0, help="only process first N (sanity)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("-o", help="output residual .mrcs")
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(0 if self_test() else 1)
    missing = [k for k in ("particles", "poses", "ctf", "consensus", "mask", "o")
               if not getattr(args, k)]
    if missing:
        ap.error("missing required args: " + ", ".join("--" + m for m in missing))
    run(args)


if __name__ == "__main__":
    main()
