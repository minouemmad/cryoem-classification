"""64 px feature mask from decoded centres: mean |volume|, 0.90 quantile, cosine dilation 6 / 6 A.
Usage: make_mask.py DECODED_DIR OUT.mrc
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cryo_utils import read_volume  # noqa: E402
from cryodrgn.masking import cosine_dilation_mask  # noqa: E402
from cryodrgn.mrcfile import write_mrc  # noqa: E402

d, out = Path(sys.argv[1]), sys.argv[2]
m = json.load(open(d / "sample.json"))
box, apix = m["box"], m["apix"]
env = np.zeros((box,) * 3)
for i in range(m["samples"]):
    env += np.abs(read_volume(d / "volumes" / f"vol_{i + 1:03d}.mrc", box, apix))
env /= m["samples"]
mask = cosine_dilation_mask(env, threshold=float(np.quantile(env, 0.90)), dilation=6, edge_dist=6, apix=apix).astype(np.float32)
write_mrc(out, mask, Apix=apix, is_vol=True)
print("mask voxels", int((mask > 0.5).sum()))
