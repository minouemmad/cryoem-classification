"""True hand (z flip) x protein mask for the aligned trajectory maps, plus difference-from-mean maps for surface
colouring. Usage: premask.py TRAJ_DIR
"""
import glob, os, sys, json
import numpy as np, mrcfile
from scipy.ndimage import gaussian_filter
L = os.path.abspath(sys.argv[1]); V7 = "/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn/J4001_gP25_WT/candidate_groups_epoch50/presentation_v7"
mask = mrcfile.read(f"{V7}/models/protein_maskT.mrc").astype(np.float32)
apix = float(mrcfile.open(f"{V7}/models/ref_true.mrc").voxel_size.x)
os.makedirs(f"{L}/maskedT", exist_ok=True)
rd = lambda n: mrcfile.read(f"{L}/aligned/{n}.mrc")[::-1].astype(np.float32) * mask
def save(a, n):
    with mrcfile.new(f"{L}/maskedT/{n}.mrc", overwrite=True) as o: o.set_data(a.astype(np.float32)); o.voxel_size = apix
names = [os.path.basename(f)[:-4] for f in sorted(glob.glob(f"{L}/aligned/lat_*.mrc")) + sorted(glob.glob(f"{L}/aligned/vol_*.mrc"))]
V = {n: rd(n) for n in names}
mean = np.mean(list(V.values()), 0); save(mean, "traj_mean")
sds = []
for n, v in V.items():
    save(v, n); d = gaussian_filter(v - mean, 1.5); save(d, n + "_diff"); sds.append(float(d[mean > np.percentile(mean, 97)].std()))
for f in sorted(glob.glob(f"{L}/aligned/latpc_*.mrc")):
    n = os.path.basename(f)[:-4]; save(rd(n), n)
json.dump(dict(diff_sd=float(np.median(sds)), apix=apix), open(f"{L}/maskedT/render.json", "w"))
print(L, "diff sd", np.median(sds))
