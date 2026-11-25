#!/usr/bin/env python
"""Denoise a particle stack with Topaz-Denoise's pretrained general 2-D model, in GPU batches
(same per-image computation as `topaz denoise`, normalisation restored per particle). Run in the `topaz` env.
Usage: topaz_denoise.py in.mrcs out.mrcs [--batch 256] [--first N]
"""
import argparse
import sys
import time

import mrcfile
import numpy as np
import torch
import os
import topaz
from topaz.denoising.models import UDenoiseNet


def load_unet():
    """Topaz's default pretrained 2-D model ('unet' = unet_L2_v0.2.2), loaded from the package file directly
    (topaz-em 0.3.20's load_model fails to resolve its own '../pretrained' resource path)."""
    model = UDenoiseNet(base_width=11, top_width=5)
    f = os.path.join(os.path.dirname(topaz.__file__), "pretrained", "denoise", "unet_L2_v0.2.2.sav")
    model.load_state_dict(torch.load(f, map_location="cpu", weights_only=False))
    return model.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inp")
    p.add_argument("out")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--first", type=int, default=0, help="only the first N particles (testing)")
    a = p.parse_args()

    model = load_unet().cuda()
    src = mrcfile.mmap(a.inp, mode="r", permissive=True)
    data = src.data
    N = len(data) if not a.first else min(a.first, len(data))
    D = data.shape[-1]
    out = mrcfile.new_mmap(a.out, shape=(N, D, D), mrc_mode=2, overwrite=True)
    out.voxel_size = src.voxel_size
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, N, a.batch):
            x = torch.from_numpy(np.asarray(data[s:s + a.batch], dtype=np.float32)).cuda()
            mu = x.mean(dim=(1, 2), keepdim=True)
            sd = x.std(dim=(1, 2), keepdim=True).clamp(min=1e-8)
            y = model(((x - mu) / sd).unsqueeze(1)).squeeze(1) * sd + mu
            out.data[s:s + len(x)] = y.cpu().numpy()
            if (s // a.batch) % 50 == 0:
                print(f"{s + len(x)}/{N}  {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    out.update_header_stats()
    out.close()
    src.close()
    print(f"done {N} particles in {time.time() - t0:.0f}s -> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
