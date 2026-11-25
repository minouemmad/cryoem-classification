"""Scale-matched difference maps between cryo-EM maps on the same grid.

B is fitted to A by least squares (A ~ a*B + c) inside a shared envelope after both are low-passed to the same
resolution, so the difference shows redistributed density rather than scale / B-factor differences.
  diff(A, B, apix, res=10, env=None) -> (A - a*B - c, a, c)
  noise_sd(pairs, ...)               -> sd of the same difference for noise pairs (thresholds)
  envelope(maps, apix)               -> soft mask from the mean of the maps
CLI: mapdiff.py OUT.mrc A.mrc B.mrc [--res 10]
"""
import sys
import numpy as np, mrcfile
from scipy.ndimage import gaussian_filter, binary_dilation


def lowpass(v, apix, res):
    """Gaussian low-pass whose transfer function is 0.5 at 1/res (real-space sigma = res * sqrt(2 ln 2) / (2 pi))."""
    return gaussian_filter(v.astype(np.float32), 0.1874 * res / apix) if res else v


def envelope(maps, apix, res=10, k=1.0, dil=3):
    m = np.mean([lowpass(x, apix, res) for x in maps], 0)
    e = binary_dilation(m > m.mean() + k * m.std(), iterations=dil)
    return gaussian_filter(e.astype(np.float32), 1.0)


def diff(A, B, apix, res=10, env=None):
    a_, b_ = lowpass(A, apix, res), lowpass(B, apix, res)
    if env is None:
        env = envelope([A, B], apix, res)
    w = env.ravel(); x = b_.ravel(); y = a_.ravel()
    X = np.stack([x, np.ones_like(x)], 1) * w[:, None]
    sc, off = np.linalg.lstsq(X, y * w, rcond=None)[0]
    return ((a_ - sc * b_ - off) * env).astype(np.float32), float(sc), float(off)


def noise_sd(pairs, apix, res=10, env=None):
    return float(np.mean([diff(a, b, apix, res, env)[0][env > 0.5].std() for a, b in pairs]))


if __name__ == "__main__":
    o, fa, fb = sys.argv[1:4]; res = float(sys.argv[sys.argv.index("--res") + 1]) if "--res" in sys.argv else 10.0
    A = mrcfile.read(fa).astype(np.float32); B = mrcfile.read(fb).astype(np.float32)
    apix = float(mrcfile.open(fa, permissive=True).voxel_size.x)
    d, sc, off = diff(A, B, apix, res)
    with mrcfile.new(o, overwrite=True) as h:
        h.set_data(d); h.voxel_size = apix
    print(f"scale {sc:.3f} offset {off:.4f} sd {d[d != 0].std():.4f}")
