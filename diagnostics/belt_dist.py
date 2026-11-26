"""(11-25) Where in the membrane belt is the class signal?  From the saved half maps (envtest2): class signal E = dA*dB outside the union envelope, inside the belt slab,
binned by distance from the union-envelope surface (A).  TM-helix / TMD-half motion should sit next to the protein surface; detergent-micelle differences further out (ring)."""
import os
import json, numpy as np
from scipy.ndimage import distance_transform_edt
B = os.environ.get("RES", "/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn") + "/model_pipeline/"; E = B + "env/"; r2 = json.load(open(B + "od/envtest/envtest2.json"))
for ds in ("J2708", "J399"):
    z = np.load(B + f"od/envtest/{ds}_halves.npz"); A, Bm, n, names = z["A"], z["B"], z["n"], list(z["names"]); w = n / n.sum(); cA = np.tensordot(w, A, 1); cB = np.tensordot(w, Bm, 1)
    ap = float(open(E + f"{ds}/apix.txt").read()); u = np.load(E + f"{ds}/env64_union.npy").astype(bool); cons = np.load(E + f"{ds}/env64.npy").astype(bool)
    dist = distance_transform_edt(~u) * ap; ax = json.load(open(E + f"{ds}/axis.json")); a = np.array(ax["axis"]); c = np.array(ax["tmd_centre"]); g = np.indices(u.shape)
    t = (g[2] - c[0]) * a[0] + (g[1] - c[1]) * a[1] + (g[0] - c[2]) * a[2]; lo, hi = r2[ds]["belt_slab"]; belt = (t >= lo - 0.5) & (t <= hi + 0.5) & ~u
    sph = ((g - 32) ** 2).sum(0) < (0.45 * 64) ** 2; bins = [0, 5, 10, 15, 20, 30, 99]
    cd = [int((belt & cons & (dist > b0) & (dist <= b1)).sum()) for b0, b1 in zip(bins, bins[1:])]
    print(f"{ds} (voxel {ap:.2f} A, belt slab {lo}..{hi} = {(hi - lo + 1) * ap:.0f} A thick) | consensus-envelope voxels in belt by distance from protein:", dict(zip([f"{b0}-{b1}" for b0, b1 in zip(bins, bins[1:])], cd)))
    for k, nm in enumerate(names):
        p = (A[k] - cA) * (Bm[k] - cB); tot = p[belt & sph].sum()
        print(f"  {nm} n={int(n[k]):6d} belt signal by distance (A): " + "  ".join(f"{b0}-{b1}: {p[belt & sph & (dist > b0) & (dist <= b1)].sum() / tot:.2f}" for b0, b1 in zip(bins, bins[1:])))
