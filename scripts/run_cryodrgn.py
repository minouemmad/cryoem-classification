#!/usr/bin/env python
"""Turnkey cryoDRGN driver for the CFTR particle stacks.

Runs the full cryoDRGN heterogeneity workflow with CFTR-appropriate defaults:

    parse_pose_csparc + parse_ctf_csparc  (from a CryoSPARC *passthrough* .cs)
    downsample                            (the raw particle images)
    train_vae                             (learn per-particle latent z)
    analyze                               (PCA/UMAP + kmeans volumes + notebooks)

Classification then comes from clustering the latent z (see
scripts/compare_cryodrgn_classification.py).

IMPORTANT — what you must supply
--------------------------------
cryoDRGN needs the raw particle *images*, which are NOT in this workspace (the .cs
files here are metadata only; the image .mrc stacks live on the CryoSPARC server,
e.g. J995/reconstructed/<uid>_particles.mrc, ~94 GB for J1442). Provide them with:

    --particles  path to the image stack (.mrcs/.star) OR the passthrough .cs that
                 carries blob/path pointers
    --datadir    directory the .cs blob paths are relative to (the CryoSPARC project
                 dir, so J995/reconstructed/* resolves)

Pose/CTF are parsed from --passthrough (default = --particles if it is a .cs).

GPU note
--------
train_vae is essentially infeasible on CPU for the full 230k stack. On a CPU-only
box use --quick (tiny box, subset, few epochs) just to validate the toolchain; run
the real job on a CUDA machine.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--particles", required=True,
                   help="Particle image stack (.mrcs/.star) or a passthrough .cs "
                        "with blob/path pointers.")
    p.add_argument("--passthrough", default=None,
                   help="CryoSPARC passthrough .cs for poses+CTF "
                        "(default: --particles if it ends in .cs).")
    p.add_argument("--datadir", default=None,
                   help="Base dir for .cs/.star relative image paths "
                        "(the CryoSPARC project directory).")
    p.add_argument("-o", "--outdir", required=True, help="Output working directory.")
    p.add_argument("-D", "--box", type=int, default=None,
                   help="Original image box size; auto-detected from the .cs blob "
                        "shape if omitted.")
    p.add_argument("--apix", type=float, default=None,
                   help="Pixel size (A/pix); auto-detected from the .cs blob if omitted.")
    p.add_argument("--downsample", type=int, default=128,
                   help="Box to downsample to for training (default 128; do a "
                        "second pass at 256 once the run looks good).")
    p.add_argument("--zdim", type=int, default=8,
                   help="Latent dimensionality (default 8).")
    p.add_argument("--epochs", "-n", type=int, default=50)
    p.add_argument("--enc-dim", type=int, default=256)
    p.add_argument("--enc-layers", type=int, default=3)
    p.add_argument("--dec-dim", type=int, default=256)
    p.add_argument("--dec-layers", type=int, default=3)
    p.add_argument("--ind", default=None,
                   help="PKL of particle indices to restrict training to.")
    p.add_argument("--no-ctf", action="store_true",
                   help="Train without CTF (only if images are already phase-flipped "
                        "/ CTF-free, e.g. the synthetic demo).")
    p.add_argument("--quick", action="store_true",
                   help="CPU smoke test: D=64, 2000-particle subset, 3 epochs, "
                        "small arch.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the commands without executing them.")
    return p.parse_args()


def detect_box_apix(cs_path: str):
    """Read box size + pixel size from a CryoSPARC .cs blob, if present."""
    box = apix = None
    try:
        cs = np.load(cs_path)
        names = cs.dtype.names or ()
        if "blob/shape" in names:
            shp = cs["blob/shape"][0]
            box = int(np.asarray(shp).ravel()[-1])
        if "blob/psize_A" in names:
            apix = float(cs["blob/psize_A"][0])
    except Exception as e:
        print(f"[run] could not auto-detect from {cs_path}: {e}")
    return box, apix


def run(cmd, dry):
    print("\n$ " + " ".join(str(c) for c in cmd))
    if dry:
        return 0
    return subprocess.call(cmd)


def cryodrgn_exe():
    # prefer the venv console script next to this interpreter
    d = os.path.dirname(sys.executable)
    for name in ("cryodrgn.exe", "cryodrgn"):
        cand = os.path.join(d, name)
        if os.path.exists(cand):
            return cand
    return "cryodrgn"


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    cd = cryodrgn_exe()

    passthrough = args.passthrough
    if passthrough is None and args.particles.endswith(".cs"):
        passthrough = args.particles
    if passthrough is None:
        print("[run] ERROR: provide --passthrough (a .cs with poses/CTF) or pass a "
              ".cs as --particles.")
        return 2

    box = args.box
    apix = args.apix
    if box is None or apix is None:
        db, da = detect_box_apix(passthrough)
        box = box or db
        apix = apix or da
    if box is None:
        print("[run] ERROR: could not determine box size; pass -D/--box.")
        return 2
    if apix is None:
        print("[run] ERROR: could not determine pixel size; pass --apix.")
        return 2
    print(f"[run] box D={box}  Apix={apix}")

    inputs = os.path.join(args.outdir, "inputs")
    os.makedirs(inputs, exist_ok=True)
    poses = os.path.join(inputs, "poses.pkl")
    ctf = os.path.join(inputs, "ctf.pkl")

    down = 64 if args.quick else args.downsample
    epochs = 3 if args.quick else args.epochs
    enc_dim = 128 if args.quick else args.enc_dim
    enc_layers = 2 if args.quick else args.enc_layers
    dec_dim = 128 if args.quick else args.dec_dim
    dec_layers = 2 if args.quick else args.dec_layers
    zdim = 4 if args.quick else args.zdim

    # 1) poses + CTF
    rc = run([cd, "parse_pose_csparc", passthrough, "-D", str(box), "-o", poses],
             args.dry_run)
    if rc:
        return rc
    rc = run([cd, "parse_ctf_csparc", passthrough, "-D", str(box),
              "--Apix", str(apix), "-o", ctf], args.dry_run)
    if rc:
        return rc

    # 2) downsample images
    stack = os.path.join(inputs, f"particles.{down}.mrcs")
    ds_cmd = [cd, "downsample", args.particles, "-D", str(down), "-o", stack]
    if args.datadir:
        ds_cmd += ["--datadir", args.datadir]
    rc = run(ds_cmd, args.dry_run)
    if rc:
        print("[run] downsample failed — this usually means the particle IMAGES are "
              "not reachable (check --particles/--datadir). The .cs metadata in this "
              "workspace does NOT contain images.")
        return rc

    # optional quick subset
    ind = args.ind
    if args.quick and ind is None and not args.dry_run:
        try:
            cs = np.load(passthrough)
            n = len(cs)
            sub = np.sort(np.random.default_rng(0).choice(
                n, size=min(2000, n), replace=False))
            ind = os.path.join(inputs, "ind.quick.pkl")
            import pickle
            with open(ind, "wb") as f:
                pickle.dump(sub, f)
            print(f"[run] quick subset: {len(sub)} particles -> {ind}")
        except Exception as e:
            print(f"[run] quick subset skipped: {e}")

    # 3) train_vae
    train_dir = os.path.join(args.outdir, "train")
    tv = [cd, "train_vae", stack, "--poses", poses,
          "--zdim", str(zdim), "-n", str(epochs),
          "--enc-dim", str(enc_dim), "--enc-layers", str(enc_layers),
          "--dec-dim", str(dec_dim), "--dec-layers", str(dec_layers),
          "-o", train_dir]
    if not args.no_ctf:
        tv += ["--ctf", ctf]
    if ind:
        tv += ["--ind", ind]
    rc = run(tv, args.dry_run)
    if rc:
        return rc

    # 4) analyze last epoch
    rc = run([cd, "analyze", train_dir, str(epochs - 1)], args.dry_run)
    if rc:
        return rc

    print("\n[run] done.")
    print(f"  latent z:   {os.path.join(train_dir, f'z.{epochs - 1}.pkl')}")
    print(f"  analysis:   {os.path.join(train_dir, f'analyze.{epochs - 1}')}")
    print("  classify:   python scripts/compare_cryodrgn_classification.py "
          f"--workdir {train_dir} --epoch {epochs - 1} --cs {args.particles} "
          f"--protein-only -o {os.path.join(args.outdir, 'comparison')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
