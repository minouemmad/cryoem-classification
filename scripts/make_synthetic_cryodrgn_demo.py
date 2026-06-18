#!/usr/bin/env python
"""Generate a synthetic two-conformation particle stack to validate the cryoDRGN
pipeline end-to-end on THIS (CPU-only) machine.

Why this exists
---------------
cryoDRGN needs raw particle *images* + a GPU to classify the real CFTR particles,
and neither is available locally (the image stacks live on the CryoSPARC server and
torch here is CPU-only). To prove the whole chain works before investing GPU time,
this script renders a small synthetic stack from two *real, structurally distinct*
CFTR class maps (J1442 class06 = P6 and class08 = P8, map-map CC ~ 0.32). If
cryoDRGN's latent space cleanly separates the two populations, the encoder/training/
analysis/comparison toolchain is validated.

Self-consistency
----------------
Each particle image is rendered as  sum_z( affine_transform(V, matrix=R) )  i.e. the
real-space projection of the conformation volume V along the rotated z-axis defined by
the rotation matrix R that we store in poses.pkl. By the Fourier-slice theorem this is
consistent with how cryoDRGN samples the central slice at R @ (kx, ky, 0), so the poses
are self-consistent with the images and the decoder can actually fit them.

No CTF is applied (images are clean projections + Gaussian noise), so train_vae is run
WITHOUT --ctf for the demo.

Outputs (default results_cryodrgn/demo_synth/inputs/):
  particles.mrcs   - (N, D, D) float32 image stack
  poses.pkl        - (rots (N,3,3) float32, trans (N,2) zeros float32)
  ctf.pkl          - (N,9) float32 (provided for completeness; df=0)
  gt_labels.pkl    - (N,) int  ground-truth conformation label (0=P6, 1=P8)
  README.txt       - how to train + compare
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
from scipy.ndimage import affine_transform, zoom

# cryodrgn ships its own MRC I/O (mrcfile package is not installed in the venv)
from cryodrgn.mrcfile import parse_mrc, write_mrc


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p.add_argument("--map-a", default=os.path.join(
        here, "data", "maps", "J1442_classes", "J1442_class06.mrc"),
        help="Conformation A volume (default J1442 class06 / P6).")
    p.add_argument("--map-b", default=os.path.join(
        here, "data", "maps", "J1442_classes", "J1442_class08.mrc"),
        help="Conformation B volume (default J1442 class08 / P8).")
    p.add_argument("-o", "--outdir", default=os.path.join(
        here, "results_cryodrgn", "demo_synth", "inputs"),
        help="Output directory for the synthetic inputs.")
    p.add_argument("-D", "--box", type=int, default=64,
                   help="Downsampled box size for the synthetic images (default 64).")
    p.add_argument("-n", "--n-per-class", type=int, default=2000,
                   help="Number of particles per conformation (default 2000).")
    p.add_argument("--snr", type=float, default=0.5,
                   help="Approximate signal-to-noise ratio (variance ratio). Lower = "
                        "noisier / harder. Default 0.5.")
    p.add_argument("--identity-poses", action="store_true",
                   help="Project every particle along +z with NO rotation (identity "
                        "poses). This is a bulletproof positive control: it removes all "
                        "pose-convention risk so any failure to separate the two "
                        "conformations is the model's, not the generator's.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_and_resize(path: str, box: int) -> np.ndarray:
    vol, _ = parse_mrc(path)
    vol = np.asarray(vol, dtype=np.float32)
    if vol.shape[0] != box:
        factor = box / vol.shape[0]
        vol = zoom(vol, factor, order=1).astype(np.float32)
    # normalize to zero mean / unit std of the foreground signal
    vol = vol - vol.mean()
    s = vol.std()
    if s > 0:
        vol = vol / s
    return vol


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniformly random 3D rotation matrix (Shoemake / quaternion method)."""
    u1, u2, u3 = rng.random(3)
    q = np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
    ])
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def project(vol: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Real-space projection of `vol` along the z-axis rotated by R.

    image = sum_z affine_transform(vol, matrix=R), rotating about the box centre.
    This matches cryoDRGN's central-slice sampling at R @ (kx, ky, 0).
    """
    d = vol.shape[0]
    c = (d - 1) / 2.0
    offset = c - R @ np.array([c, c, c], dtype=np.float32)
    rotated = affine_transform(vol, matrix=R, offset=offset, order=1,
                               mode="constant", cval=0.0)
    return rotated.sum(axis=0).astype(np.float32)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    print(f"[demo] loading + resizing maps to D={args.box}")
    vol_a = load_and_resize(args.map_a, args.box)
    vol_b = load_and_resize(args.map_b, args.box)
    print(f"[demo] map A {os.path.basename(args.map_a)}  map B {os.path.basename(args.map_b)}")

    n = args.n_per_class
    N = 2 * n
    D = args.box

    # Share the SAME set of orientations between both conformations so the only
    # systematic difference between the two image populations is the structure.
    if args.identity_poses:
        rots = np.tile(np.eye(3, dtype=np.float32), (N, 1, 1))
    else:
        rot_half = np.stack([random_rotation(rng) for _ in range(n)]).astype(np.float32)
        rots = np.concatenate([rot_half, rot_half], axis=0)  # reuse for both classes
    labels = np.concatenate([np.zeros(n, int), np.ones(n, int)])

    mode = "identity (z-projection only)" if args.identity_poses else "random orientations"
    print(f"[demo] rendering {N} projections (D={D}, poses: {mode}) ...")
    images = np.empty((N, D, D), dtype=np.float32)
    for i in range(N):
        vol = vol_a if labels[i] == 0 else vol_b
        images[i] = project(vol, rots[i])
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{N}")

    # Add Gaussian noise scaled to the requested SNR (signal variance / noise variance)
    sig_var = images.var()
    noise_var = sig_var / max(args.snr, 1e-6)
    images = images + rng.normal(0.0, np.sqrt(noise_var), size=images.shape).astype(np.float32)

    # Shuffle so class order is not trivially recoverable from row index
    perm = rng.permutation(N)
    images = images[perm]
    rots = rots[perm]
    labels = labels[perm]

    trans = np.zeros((N, 2), dtype=np.float32)

    # ctf.pkl: (N,9) = [D, Apix, df1, df2, dfang, kV, Cs, w, phase]. df=0 (no CTF).
    apix = 1.107  # nominal; irrelevant for the demo
    ctf = np.zeros((N, 9), dtype=np.float32)
    ctf[:, 0] = D
    ctf[:, 1] = apix
    ctf[:, 5] = 300.0
    ctf[:, 6] = 2.7
    ctf[:, 7] = 0.1

    out = args.outdir
    write_mrc(os.path.join(out, "particles.mrcs"), images.astype(np.float32),
              is_vol=False)
    with open(os.path.join(out, "poses.pkl"), "wb") as f:
        pickle.dump((rots, trans), f)
    with open(os.path.join(out, "ctf.pkl"), "wb") as f:
        pickle.dump(ctf, f)
    with open(os.path.join(out, "gt_labels.pkl"), "wb") as f:
        pickle.dump(labels, f)

    train_cmd = (
        f"cryodrgn train_vae {os.path.join(out, 'particles.mrcs')} "
        f"--poses {os.path.join(out, 'poses.pkl')} "
        f"--zdim 4 -n 20 --enc-dim 128 --enc-layers 2 --dec-dim 128 --dec-layers 2 "
        f"--uninvert-data=False "
        f"-o {os.path.join(os.path.dirname(out), 'train')}"
    )
    readme = (
        "Synthetic two-conformation cryoDRGN validation set\n"
        "==================================================\n"
        f"maps: A={args.map_a}\n      B={args.map_b}\n"
        f"N={N} particles ({n} per conformation), D={D}, SNR~{args.snr}\n"
        "NO CTF applied (clean projections + Gaussian noise) -> train WITHOUT --ctf.\n\n"
        "Train (CPU, a few minutes):\n"
        f"  {train_cmd}\n\n"
        "Analyze:\n"
        f"  cryodrgn analyze {os.path.join(os.path.dirname(out), 'train')} 19\n\n"
        "Compare latent clusters to ground truth:\n"
        f"  python scripts/compare_cryodrgn_classification.py \\\n"
        f"    --workdir {os.path.join(os.path.dirname(out), 'train')} --epoch 19 \\\n"
        f"    --gt-labels {os.path.join(out, 'gt_labels.pkl')} --k 2 \\\n"
        f"    -o {os.path.join(os.path.dirname(out), 'comparison')}\n"
    )
    with open(os.path.join(out, "README.txt"), "w") as f:
        f.write(readme)

    print(f"\n[demo] wrote synthetic inputs to {out}")
    print(f"  particles.mrcs  {images.shape}")
    print(f"  poses.pkl  rots {rots.shape}  trans {trans.shape}")
    print(f"  ctf.pkl  {ctf.shape}   gt_labels.pkl  {labels.shape}")
    print("\nNext:\n  " + train_cmd)


if __name__ == "__main__":
    sys.exit(main())
