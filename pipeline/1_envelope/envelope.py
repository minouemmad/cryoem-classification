"""Label-free protein envelope: consensus back-projection of all particles (fixed NU poses), Fourier-cropped to 64^3 and low-passed
to 10 A as in the pipeline; envelope = largest connected region above solvent mean + 3 SD (solvent = voxels beyond 0.42 box), dilated 2 voxels.
Usage: envelope.py STACK POSES CTF OUTDIR   -> env64.npy (bool), mask64.mrc, consensus64.npy"""
import os, sys
from pathlib import Path
import numpy as np
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
from fastbp import FastBP, lowpass
from cryodrgn.mrcfile import write_mrc

st, po, ct, O = sys.argv[1:5]; os.makedirs(O, exist_ok=True)
bp = FastBP(st, po, ct); ap = bp.apix * 2; a, b = bp.halves(np.arange(bp.data.N), seed=0)
f = np.fft.fftshift(np.fft.fftn((a + b) / 2))[32:96, 32:96, 32:96]
m = lowpass(np.real(np.fft.ifftn(np.fft.ifftshift(f))).astype(np.float32) / 8, ap, 10.0)
r = np.sqrt(((np.indices(m.shape) - 31.5) ** 2).sum(0)) / 64; sol = m[r > 0.42]
lab, _ = ndimage.label(m > sol.mean() + 3 * sol.std()); e = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
e = ndimage.binary_dilation(e, iterations=2)
np.save(O + "/env64.npy", e); np.save(O + "/consensus64.npy", m); write_mrc(O + "/mask64.mrc", e.astype(np.float32), Apix=float(ap))
print("ENV", O, "apix64", ap, "voxels", int(e.sum()), "of", e.size, flush=True)
