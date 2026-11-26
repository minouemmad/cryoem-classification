"""(11-25) P7 map check on one scale, as od_compare: 10 A low-pass, protein mask, every map linearly fitted to the consensus, d2 = |x - y|^2 / |largest ref pair|^2."""
import os
import json, sys, numpy as np, mrcfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../lib" if "pipeline" in __file__ else "../lib")); from fastbp import lowpass
RC = os.environ.get("RES", "/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn") + "/"; MR = RC + "J4001_gP25_WT/method_rank/fits/"; OD = RC + "model_pipeline/od/"
mask = mrcfile.read(RC + "J4001_gP25_WT/candidate_groups_epoch50/presentation_v7/models/protein_maskT.mrc") > 0.5; ap = float(mrcfile.open(MR + "in_J1049.mrc").voxel_size.x)
L = lambda f: lowpass(mrcfile.read(f).astype(np.float32), ap, 10.0)[mask].astype(np.float64); c0 = L(MR + "in_J1049.mrc")
def fit(v): a, b = np.polyfit(v, c0, 1); return a * v + b
ref = {k: fit(L(MR + f)) for k, f in (("P8", "fit_J2975.mrc"), ("P7", "fit_J2974.mrc"), ("P6", "fit_J2973.mrc"), ("Alt0", "fit_Alt0.mrc"), ("Alt1", "fit_Alt1.mrc"))}
ks = list(ref); un = max(((ref[a] - ref[b]) ** 2).sum() for a in ks for b in ks); d2 = lambda x, y: float(((x - y) ** 2).sum() / un)
print("ref d2 to P7:", {k: round(d2(v, ref["P7"]), 3) for k, v in ref.items() if k != "P7"}, "| consensus", round(d2(c0, ref["P7"]), 3), "| P8-P6", round(d2(ref["P8"], ref["P6"]), 3))
# best possible blend of P8 / P6 (and consensus) -> how much of P7 is its own
for nm, cols in (("a P8 + b P6", ["P8", "P6"]), ("a P8 + b P6 + c Alt0", ["P8", "P6", "Alt0"])):
    X = np.stack([ref[c] for c in cols] + [np.ones_like(c0)], 1); w, *_ = np.linalg.lstsq(X, ref["P7"], rcond=None); print(f"best {nm}: weights {np.round(w[:-1], 2)} -> d2 to P7 {d2(X @ w, ref['P7']):.3f}")
rows = [r for r in json.load(open(OD + "cmp/compare_J4001.json"))["rows"] if r["run"] != "hetero"]
out = sorted((d2(fit(L(r["path"])), ref["P7"]), r["name"], r["n"], min(((d2(fit(L(r["path"])), v), k) for k, v in ref.items()))[1]) for r in rows)
print("closest class maps to P7 (d2, map, particles, nearest reference):"); [print("   %.3f  %-24s %6d  %s" % v) for v in out[:10]]
