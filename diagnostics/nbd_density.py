"""(11-26) Per-domain density in every refined J4001 map (compare frame).  Domain masks from the fitted model D1557_fitT.pdb (atoms within 4 A, residue ranges
TMD1 1-380, NBD1 381-645, R 646-839, TMD2 840-1172, NBD2 1173-1480).  Each map: 10 A lowpass, linear fit to the consensus over the envelope (as od_compare), then
domain occupancy = mean(map in domain) / mean(consensus in domain).  Refs (P8/P7/P6) are scored identically, EVALUATION ONLY.  Out: od/nbd_density.json"""
import os
import json, sys, numpy as np, mrcfile
from scipy.ndimage import distance_transform_edt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../lib" if "pipeline" in __file__ else "../lib")); from mapdiff import envelope, lowpass
R = os.environ.get("RES", "/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn") + "/"; V7 = R + "J4001_gP25_WT/candidate_groups_epoch50/presentation_v7/"
d = json.load(open(R + "model_pipeline/od/cmp/compare_J4001.json")); MR = R + "J4001_gP25_WT/method_rank/"
cons = MR + "fits/in_J1049.mrc"; REF = {"P8": MR + "fits/fit_J2975.mrc", "P7": MR + "fits/fit_J2974.mrc", "P6": MR + "fits/fit_J2973.mrc"}
m = mrcfile.open(cons); ap = float(m.voxel_size.x); org = np.array([m.header.origin.x, m.header.origin.y, m.header.origin.z], float); sh = m.data.shape
DOM = {"TMD1": (1, 380), "NBD1": (381, 645), "TMD2": (840, 1172), "NBD2": (1173, 1480)}
xyz, res, ch = [], [], set()
for l in open(V7 + "models/D1557_fitT.pdb"):
    if l.startswith(("ATOM", "HETATM")) and l[21] == "A": xyz.append([float(l[30:38]), float(l[38:46]), float(l[46:54])]); res.append(int(l[22:26])); ch.add(l[21])
xyz = np.array(xyz); res = np.array(res); print("chains", sorted(ch), "res", res.min(), res.max(), "map", sh, ap, org)
ijk = np.round((xyz - org) / ap).astype(int)[:, ::-1]                      # z,y,x index order
lab = np.zeros(sh, np.int8); near = np.full(sh, np.inf)
for n, (dn, (a, b)) in enumerate(DOM.items(), 1):
    seed = np.ones(sh, bool); s = ijk[(res >= a) & (res <= b)]; s = s[(s >= 0).all(1) & (s < sh[0]).all(1)]
    if not len(s): print("no atoms", dn); continue
    seed[s[:, 0], s[:, 1], s[:, 2]] = False; dt = distance_transform_edt(seed) * ap; upd = (dt < 4.0) & (dt < near); lab[upd] = n; near = np.minimum(near, dt)
L = lambda f: lowpass(mrcfile.read(f).astype(np.float32), ap, 10.0)
c0f = L(cons); env = envelope([c0f], ap, None, k=1.0, dil=3) > 0.5; c0 = c0f[env].astype(np.float64)
def fitc(v):
    x = v[env].astype(np.float64); a, b = np.polyfit(x, c0, 1); out = np.zeros(sh); out[env] = a * x + b; return out
cf = np.zeros(sh); cf[env] = c0; base = {dn: cf[(lab == n) & env].mean() for n, dn in enumerate(DOM, 1)}
print("voxels", {dn: int(((lab == n) & env).sum()) for n, dn in enumerate(DOM, 1)})
occ = lambda v: {dn: float(v[(lab == n) & env].mean() / base[dn]) for n, dn in enumerate(DOM, 1)}
out = dict(refs={k: occ(fitc(L(f))) for k, f in REF.items()}, rows=[])
for k, v in out["refs"].items(): print(f"REF {k:4s}", " ".join(f"{dn} {x:.2f}" for dn, x in v.items()))
for r in d["rows"]:
    if r["run"] == "hetero": continue
    o = occ(fitc(L(r["path"]))); out["rows"].append(dict(name=r["name"], run=r["run"], K=r["K"], n=r["n"], closest=r["closest"], d2=r["d2"], occ=o))
json.dump(out, open(R + "model_pipeline/od/nbd_density.json", "w"), indent=1)
rows = sorted(out["rows"], key=lambda r: r["occ"]["NBD1"])
print("lowest NBD1:"); [print(f"  {r['name']:28s} n {r['n']:6d} near {r['closest']:4s} d2P8 {r['d2']['P8']:.2f}", " ".join(f"{dn} {x:.2f}" for dn, x in r["occ"].items())) for r in rows[:12]]
p8 = [r for r in out["rows"] if r["closest"] == "P8"]; print("nearest-P8 maps:", len(p8), "mean NBD1 occ", np.mean([r["occ"]["NBD1"] for r in p8]).round(3) if p8 else None)
for r in sorted(p8, key=lambda r: r["d2"]["P8"])[:6]: print(f"  P8-like {r['name']:28s} n {r['n']:6d} d2P8 {r['d2']['P8']:.2f}", " ".join(f"{dn} {x:.2f}" for dn, x in r["occ"].items()))
x = np.array([[r["d2"]["P8"] - r["d2"]["P6"], r["occ"]["NBD1"]] for r in out["rows"]]); print("corr(P8-ward, NBD1 occ) =", np.corrcoef(-x[:, 0], x[:, 1])[0, 1].round(3))
