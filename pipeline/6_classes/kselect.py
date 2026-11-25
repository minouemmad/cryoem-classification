"""Label-free class-number scan, K = 2..8, for one pipeline run (no heterorefinement input).  Per K:
  gain     relative drop of the within-class map variance from K-1 to K;  own = smallest own (off-blend) share (MP.json min_offblend)
  sh_min   smallest split-half cc of a class map (A-half vs B-half cell maps, class minus mean)
  own_sh   smallest split-half cc of a class's own component;  minfrac = smallest class / all particles
  seeds    (select/repro_<DS>.json, when seed runs exist) classes reproduced / distinct across seeds
  pass     sh_min >= 0.9, own_sh >= 0.5 (K > 2), minfrac >= 0.04, and all classes distinct across seeds when seeds exist
Candidates to refine = every passing K + the elbow + (largest passing K + 1) as a boundary check.
Usage: kselect.py DS [RUN=rae_base]   -> $WORK/select/kselect_<DS>.json"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
from maps import cc, class_maps, own_component

ds = sys.argv[1]; run = sys.argv[2] if len(sys.argv) > 2 else "rae_base"; d = os.environ["WORK"] + f"/pipe/{ds}/{run}/mc/"; O = os.environ["WORK"] + "/select/"; os.makedirs(O, exist_ok=True)
mj = json.load(open(d + "MP.json"))["model"]; z, c = np.load(d + "MP_model_labels.npz"), np.load(d + "MP_model_cells.npz")
n = np.bincount(c["cell"], minlength=len(z["K2"])).astype(float); A, B = c["A"].astype(np.float64), c["B"].astype(np.float64)
mean = lambda X: (X * n[:, None]).sum(0) / n.sum()
rep = json.load(open(O + f"repro_{ds}.json")) if os.path.exists(O + f"repro_{ds}.json") else None; rr = {r["K"]: r for r in rep["rows"]} if rep else {}
W = mj["within"]; rows = []
print(f"{ds} {run}: elbow {mj['elbow']}, cv_min {mj['cv_min']}, off-blend halving {mj['offblend_halving']}, seeds {rep['seeds'] if rep else 'none yet'}")
print("  K  gain   own   sh_min own_sh minfrac  seeds rep/dist  pass")
for K in range(2, 9):
    lab = z[f"K{K}"]; RA, RB = class_maps(A, n, lab, K), class_maps(B, n, lab, K); MA, MB = RA - mean(A), RB - mean(B)
    sh = min(cc(MA[k], MB[k]) for k in range(K)); osh = min(cc(own_component(RA, k), own_component(RB, k)) for k in range(K)) if K > 2 else float("nan")
    sz = np.array([n[lab == k].sum() for k in range(K)]); mf = float(sz.min() / sz.sum()); gain = (W[K - 2] - W[K - 1]) / W[K - 2]
    own = float(mj["min_offblend"].get(str(K), float("nan"))); s = rr.get(K)
    ok = sh >= 0.9 and (K == 2 or osh >= 0.5) and mf >= 0.04 and (s is None or s["n_distinct"] == K)
    rows.append(dict(K=K, gain=gain, own=own, sh_min=sh, own_sh=osh, minfrac=mf, sizes=sorted(sz.tolist(), reverse=True), seeds_rep=s and s["n_reproducible"], seeds_distinct=s and s["n_distinct"], passes=bool(ok)))
    print(f"  {K}  {gain:5.3f}  {own:5.3f}  {sh:5.2f}  {osh:5.2f}  {mf:6.3f}   {'' if s is None else str(s['n_reproducible']) + '/' + str(s['n_distinct']):>9}       {'yes' if ok else 'no'}")
ps = [r["K"] for r in rows if r["passes"]]; cand = sorted(set(ps + [mj["elbow"]] + ([max(ps) + 1] if ps and max(ps) < 8 else [])))
print(f"  passing K: {ps}; candidates to refine: {cand}")
json.dump(dict(ds=ds, run=run, elbow=mj["elbow"], cv_min=mj["cv_min"], rows=rows, passing=ps, candidates=cand), open(O + f"kselect_{ds}.json", "w"), indent=1)
