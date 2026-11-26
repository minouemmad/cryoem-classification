"""(11-25) Start dependence: for each dataset and K, refined class maps of every pair of starts (same aligned frame as od_compare: fit_*.mrc), difference from the consensus
(mean of the heterorefinement fits) inside the protein mask at 10 A, Hungarian match, mean matched cc.  Output od/xstart.json + table."""
import os
import json, itertools, numpy as np, mrcfile
from scipy.optimize import linear_sum_assignment
B = os.environ.get("RES", "/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn") + "/"; O = B + "model_pipeline/od/"
mask = mrcfile.read(B + "J4001_gP25_WT/candidate_groups_epoch50/presentation_v7/models/protein_maskT.mrc") > 0.5
ST = ["start_free", "start_vq", "start_mlp", "start_diffmap", "start_latent"]; out = {}
for ds, Ks in (("J4001", range(2, 6)), ("J2708", range(2, 6)), ("J399", range(2, 11)), ("J207", range(2, 6))):
    d = json.load(open(O + f"cmp/compare_{ds}.json")); rows = d["rows"]
    het = [mrcfile.read(r["path"])[mask] for r in rows if r["run"] == "hetero"]; cons = np.mean(het, 0)
    vec = lambda r: mrcfile.read(r["path"])[mask] - cons
    out[ds] = {}
    for K in Ks:
        sets = {s: [r for r in sorted(rows, key=lambda r: r["k"]) if r["run"] == s and r["K"] == K] for s in ST}
        sets = {s: [vec(r) for r in v] for s, v in sets.items() if len(v) == K}
        for a, b in itertools.combinations(sets, 2):
            C = np.array([[np.corrcoef(x, y)[0, 1] for y in sets[b]] for x in sets[a]]); i, j = linear_sum_assignment(-C)
            out[ds][f"K{K} {a[6:]}-{b[6:]}"] = dict(mean=round(float(C[i, j].mean()), 3), min=round(float(C[i, j].min()), 3))
json.dump(out, open(O + "xstart.json", "w"), indent=1)
for ds, v in out.items():
    print(ds); [print(f"  {k:24s} mean {x['mean']:.2f}  min {x['min']:.2f}") for k, x in v.items()]
