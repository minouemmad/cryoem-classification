#!/usr/bin/env python
"""Run the full cryoDRGN heterogeneity workflow on a CryoSPARC particle stack.

parse poses/CTF -> downsample images -> train_vae -> analyze. Classification comes
from clustering the latent z (see cryodrgn_compare.py).

Requires the raw particle IMAGES (not just .cs metadata): pass the image stack or a
.cs with blob/path pointers via --particles, and --datadir so those paths resolve.
Full-size training needs a GPU; use --quick for a CPU smoke test.
"""
from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--particles", required=True,
                   help="Image stack (.mrcs/.star) or .cs with blob/path pointers.")
    p.add_argument("--passthrough", help="CryoSPARC .cs for poses+CTF "
                   "(default: --particles if it is a .cs).")
    p.add_argument("--datadir", help="Base dir for relative image paths.")
    p.add_argument("-o", "--outdir", required=True)
    p.add_argument("-D", "--box", type=int, help="Image box (auto from .cs blob).")
    p.add_argument("--apix", type=float, help="Pixel size (auto from .cs blob).")
    p.add_argument("--downsample", type=int, default=128)
    p.add_argument("--zdim", type=int, default=8)
    p.add_argument("--epochs", "-n", type=int, default=50)
    p.add_argument("--enc-dim", type=int, default=256)
    p.add_argument("--enc-layers", type=int, default=3)
    p.add_argument("--dec-dim", type=int, default=256)
    p.add_argument("--dec-layers", type=int, default=3)
    p.add_argument("--ind", help="PKL of particle indices to train on.")
    p.add_argument("--no-ctf", action="store_true")
    p.add_argument("--quick", action="store_true",
                   help="CPU smoke test: D=64, 2000-particle subset, 3 epochs.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def detect_box_apix(cs_path):
    cs = np.load(cs_path)
    names = cs.dtype.names or ()
    box = int(np.asarray(cs["blob/shape"][0]).ravel()[-1]) if "blob/shape" in names else None
    apix = float(cs["blob/psize_A"][0]) if "blob/psize_A" in names else None
    return box, apix


def run(cmd, dry):
    print("\n$ " + " ".join(map(str, cmd)))
    return 0 if dry else subprocess.call(cmd)


def cryodrgn_exe():
    d = os.path.dirname(sys.executable)
    for name in ("cryodrgn.exe", "cryodrgn"):
        if os.path.exists(os.path.join(d, name)):
            return os.path.join(d, name)
    return "cryodrgn"


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    cd = cryodrgn_exe()

    passthrough = args.passthrough or (args.particles if args.particles.endswith(".cs") else None)
    if passthrough is None:
        sys.exit("ERROR: provide --passthrough (.cs with poses/CTF) or a .cs as --particles.")

    box, apix = args.box, args.apix
    if box is None or apix is None:
        db, da = detect_box_apix(passthrough)
        box, apix = box or db, apix or da
    if box is None or apix is None:
        sys.exit("ERROR: could not determine box/Apix; pass -D and --apix.")
    print(f"[run] box D={box}  Apix={apix}")

    inputs = os.path.join(args.outdir, "inputs")
    os.makedirs(inputs, exist_ok=True)
    poses = os.path.join(inputs, "poses.pkl")
    ctf = os.path.join(inputs, "ctf.pkl")

    down = 64 if args.quick else args.downsample
    epochs = 3 if args.quick else args.epochs
    enc_dim, dec_dim = (128, 128) if args.quick else (args.enc_dim, args.dec_dim)
    enc_layers, dec_layers = (2, 2) if args.quick else (args.enc_layers, args.dec_layers)
    zdim = 4 if args.quick else args.zdim

    if run([cd, "parse_pose_csparc", passthrough, "-D", str(box), "-o", poses], args.dry_run):
        sys.exit(1)
    if run([cd, "parse_ctf_csparc", passthrough, "-D", str(box), "--Apix", str(apix),
            "-o", ctf], args.dry_run):
        sys.exit(1)

    stack = os.path.join(inputs, f"particles.{down}.mrcs")
    ds = [cd, "downsample", args.particles, "-D", str(down), "-o", stack]
    if args.datadir:
        ds += ["--datadir", args.datadir]
    if run(ds, args.dry_run):
        sys.exit("[run] downsample failed — particle IMAGES not reachable "
                 "(check --particles/--datadir).")

    ind = args.ind
    if args.quick and ind is None and not args.dry_run:
        n = len(np.load(passthrough))
        sub = np.sort(np.random.default_rng(0).choice(n, min(2000, n), replace=False))
        ind = os.path.join(inputs, "ind.quick.pkl")
        with open(ind, "wb") as f:
            pickle.dump(sub, f)

    train_dir = os.path.join(args.outdir, "train")
    tv = [cd, "train_vae", stack, "--poses", poses, "--zdim", str(zdim), "-n", str(epochs),
          "--enc-dim", str(enc_dim), "--enc-layers", str(enc_layers),
          "--dec-dim", str(dec_dim), "--dec-layers", str(dec_layers), "-o", train_dir]
    if not args.no_ctf:
        tv += ["--ctf", ctf]
    if ind:
        tv += ["--ind", ind]
    if run(tv, args.dry_run):
        sys.exit(1)

    if run([cd, "analyze", train_dir, str(epochs - 1)], args.dry_run):
        sys.exit(1)

    print(f"\n[run] done. latent z: {train_dir}/z.{epochs - 1}.pkl")
    print(f"  classify: python scripts/cryodrgn_compare.py --workdir {train_dir} "
          f"--epoch {epochs - 1} --cs {args.particles} --protein-only "
          f"-o {os.path.join(args.outdir, 'comparison')}")


if __name__ == "__main__":
    main()
