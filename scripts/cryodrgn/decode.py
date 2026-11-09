#!/usr/bin/env python
"""Decode a cryoDRGN run at --samples representative particles (medoids of MiniBatchKMeans clusters of z).
Writes centers.txt, center_rows.npy, volumes/vol_NNN.mrc (cryodrgn eval_vol) and sample.json.
Usage: decode.py --run RUN_DIR [--epoch 50 --samples 2000 --box 64 --native-apix 2.075 --device 0] -o OUT
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import subprocess

import numpy as np
import yaml

from cryo_utils import fingerprint, fresh, load_latents, read_volume, sample_latents, write_json

LOG = logging.getLogger("decode")
CRYODRGN = "/home/mae2183/miniconda3/envs/cryodrgn/bin/cryodrgn"


def decode(args) -> None:
    run = args.run.resolve()
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    if cfg["dataset_args"].get("ind") is not None:
        raise ValueError("Requires a fullset run without --ind")
    z = load_latents(run / f"z.{args.epoch}.pkl")
    native = int(cfg["lattice_args"]["D"]) - 1
    if args.box % 2 or args.box > native:
        raise ValueError("--box must be even and <= native box")
    rows, counts = sample_latents(z, args.samples, args.seed)
    out = fresh(args.outdir)
    np.savetxt(out / "centers.txt", z[rows], fmt="%.17g")
    np.save(out / "center_rows.npy", rows)
    weights = run / f"weights.{args.epoch}.pkl"
    apix = args.native_apix * native / args.box
    cmd = [CRYODRGN, "eval_vol", str(weights), "-c", str(run / "config.yaml"),
           "-o", str(out / "volumes"), "--zfile", str(out / "centers.txt"),
           "--Apix", f"{apix:.10g}", "--device", str(args.device)]
    if args.box != native:
        cmd += ["-d", str(args.box)]
    LOG.info("Decoding %d centres of %s", len(rows), run.name)
    subprocess.run(cmd, check=True)
    for i in range(len(rows)):
        read_volume(out / "volumes" / f"vol_{i + 1:03d}.mrc", args.box, apix)
    write_json(out / "sample.json", dict(
        run=str(run), epoch=args.epoch, n=len(z), zdim=z.shape[1], samples=len(rows),
        box=args.box, apix=apix, native_box=native, seed=args.seed,
        center_counts=counts.tolist(), z=fingerprint(run / f"z.{args.epoch}.pkl"),
        weights=fingerprint(weights)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--epoch", type=int, default=50)
    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--box", type=int, default=64)
    p.add_argument("--native-apix", type=float, default=2.075)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("-o", "--outdir", type=Path, required=True)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    decode(args)


if __name__ == "__main__":
    main()
