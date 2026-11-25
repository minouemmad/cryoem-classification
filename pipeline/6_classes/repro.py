"""Seed reproducibility of the classes and of the search rounds: the reference run pipe/<DS>/<REF> against every finished pipe/<DS>/<SEEDS>*
(independently trained models, same data).
 classes (K = 2..8): class maps (cell half maps, n-weighted, minus the mean) Hungarian-matched to the reference by cc;
          reproducible = matched cc >= 0.8 in most seeds;  distinct = own (off-blend) component cross-run cc >= 0.5.
          K_rep = largest K with all classes reproducible;  K_distinct = largest K with all classes reproducible and distinct.
 rounds:  for each reference round, best |cc| of its mode map with the seed's mode maps (median over seeds); keep rounds while >= 0.75.
Usage: repro.py DS [REF=rae_base] [SEEDS=rae_s]   -> $WORK/select/repro_<DS>.json"""
import glob, json, os, sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
from maps import cc, class_maps, own_component

ds = sys.argv[1]; REF = sys.argv[2] if len(sys.argv) > 2 else "rae_base"; SEEDS = sys.argv[3] if len(sys.argv) > 3 else "rae_s"
P = os.environ["WORK"] + f"/pipe/{ds}/"; O = os.environ["WORK"] + "/select/"; os.makedirs(O, exist_ok=True); TH, OBT, RT = 0.8, 0.5, 0.75


def load(d):
    pre = d + "/mc/MP_model"
    if not (os.path.exists(d + "/DONE") and os.path.exists(pre + "_labels.npz")): return None
    z, c = np.load(pre + "_labels.npz"), np.load(pre + "_cells.npz")
    return dict(z=z, n=np.bincount(c["cell"], minlength=len(z["K2"])).astype(float), A=c["A"].astype(np.float64), B=c["B"].astype(np.float64), dir=d)


def centred(r, K, X):
    X0 = (r["A"] + r["B"]) / 2; return class_maps(X, r["n"], r["z"][f"K{K}"], K) - (X0 * r["n"][:, None]).sum(0) / r["n"].sum()


def modes(d):
    return [np.load(f).ravel() for f in sorted(glob.glob(d + "/iter2/mode_r*.npy"), key=lambda f: int(f.split("_r")[-1][:-4]))]


ref = load(P + REF); seeds = [(os.path.basename(d), load(d)) for d in sorted(glob.glob(P + SEEDS + "*"))]; seeds = [(nm, r) for nm, r in seeds if r is not None]
if ref is None or not seeds: sys.exit(f"{ds}: needs {REF} and at least one finished {SEEDS}* run")
rows = []
for K in range(2, 9):
    Xr = (ref["A"] + ref["B"]) / 2; M0 = centred(ref, K, Xr); n0 = np.array([ref["n"][ref["z"][f"K{K}"] == k].sum() for k in range(K)])
    MA, MB = centred(ref, K, ref["A"]), centred(ref, K, ref["B"]); sh = [cc(MA[k], MB[k]) for k in range(K)]; R0 = class_maps(Xr, ref["n"], ref["z"][f"K{K}"], K); per, obx = [], []
    for nm, r in seeds:
        Xs = (r["A"] + r["B"]) / 2; M1 = centred(r, K, Xs); C = np.array([[cc(a, b) for b in M1] for a in M0]); i, j = linear_sum_assignment(-C); m = np.empty(K); m[i] = C[i, j]; per.append(m)
        R1 = class_maps(Xs, r["n"], r["z"][f"K{K}"], K); mt = dict(zip(i, j)); obx.append([cc(own_component(R0, k), own_component(R1, mt[k])) for k in range(K)])
    per, obx = np.array(per), np.median(np.array(obx), 0); med = np.median(per, 0); frac = (per >= TH).mean(0); o = np.argsort(-n0)
    nrep = int((frac > 0.5).sum()); ndis = int(sum(frac[k] > 0.5 and obx[k] >= OBT for k in range(K)))
    rows.append(dict(K=K, sizes=n0[o].tolist(), split_half=[sh[k] for k in o], matched_median=[float(med[k]) for k in o], offblend_cross_run=[float(obx[k]) for k in o],
                     n_reproducible=nrep, n_distinct=ndis, per_seed={nm: per[q][o].tolist() for q, (nm, _) in enumerate(seeds)}))
    print(f"K {K}: sizes % {' '.join(f'{100 * x / n0.sum():4.0f}' for x in n0[o])} | split-half {' '.join(f'{sh[k]:.2f}' for k in o)} | matched {' '.join(f'{med[k]:.2f}' for k in o)}"
          f" | own cross-run {' '.join(f'{obx[k]:.2f}' for k in o)} -> reproduced {nrep}/{K}, distinct {ndis}/{K}")
Krep = max([r["K"] for r in rows if r["n_reproducible"] == r["K"]] or [1]); Kdis = max([r["K"] for r in rows if r["n_distinct"] == r["K"]] or [1])
ma = modes(ref["dir"]); rr = [[max(abs(cc(x, y)) for y in mb) if mb else np.nan for x in ma] for mb in (modes(r["dir"]) for _, r in seeds)]
rmed = np.nanmedian(np.array(rr), 0) if rr else np.array([]); keep = next((i for i, v in enumerate(rmed) if v < RT), len(rmed))
print(f"rounds (median best |r| with the seeds): {' '.join(f'{v:.2f}' for v in rmed)} -> keep {keep}; K_rep {Krep}, K_distinct {Kdis}")
json.dump(dict(ds=ds, reference=REF, seeds=[nm for nm, _ in seeds], rows=rows, K_rep=Krep, K_distinct=Kdis, round_r=rmed.tolist(), keep_rounds=keep), open(O + f"repro_{ds}.json", "w"), indent=1)
