"""Are the window maps mixtures? Each map x (10 A, envelope of the pipeline's maps, scale/offset-matched) is fitted as x ~ a P8 + b P6 + c P7 + e with the best P8 / P6 / P7 maps (J2975 / J2973 / J2974; r
Usage: state_mixture_fit.py -> p7_unmix_1119.log"""
import json, sys
from pathlib import Path
import numpy as np, mrcfile
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from mapdiff import envelope, lowpass
H = Path("/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn/J4001_gP25_WT/method_rank"); F = H / "fits"; AS = H.parent / "candidate_groups_epoch50/axis_survey"
apix = float(mrcfile.open(F / "fit_J2307.mrc").voxel_size.x); rd = lambda n: mrcfile.read(F / (f"in_{n}.mrc" if n == "J1049" else f"fit_{n}.mrc")).astype(np.float32)
rows = json.load(open(H / "expl_report.json")); lab = np.load(H / "j2923_labels.npy")
G = {}
for f in AS.glob("cryosparc_jobs*.json"):
    try: d = json.load(open(f))
    except Exception: continue
    for k, v in d.items():
        if isinstance(v, dict) and v.get("nu") and v.get("group"):
            key = ("P17_" if v.get("project") == "P17" else "") + v["nu"]; parts = v["group"].split("/")
            if len(parts) >= 2: G.setdefault(key, AS / parts[-2] / "groups" / (parts[-1] + ".npy"))
P = {k: lowpass(rd(j), apix, 10) for k, j in (("P8", "J2975"), ("P6", "J2973"), ("P7", "J2974"), ("cons", "J1049"), ("e8", "J2307"), ("e6", "J2334"), ("pick", "J2384"), ("r1", "J1704"), ("r2", "J1707"))}
envp = envelope([P["pick"], P["e8"], P["e6"], P["cons"]], apix, None, k=1.0, dil=3) > 0.5
mean3 = np.mean([P[k][envp] for k in ("pick", "e8", "e6")], 0); nm = lambda v: (lambda ab: ab[0] * v + ab[1])(np.polyfit(v, mean3, 1))
X = {k: nm(v[envp].astype(np.float64)) for k, v in P.items()}
B = np.stack([X["P8"], X["P6"], X["P7"], np.ones_like(X["P8"])], 1); un = (X["e6"] - X["e8"]) @ (X["e6"] - X["e8"])
def fit(x):
    co = np.linalg.lstsq(B, x, rcond=None)[0]; r = x - B @ co; return co, float(r @ r / un)
print("fit x = a P8 + b P6 + c P7 + e  (c = P7 content; residual in units of |P6-P8|^2)")
for nme, k in (("best P8", "P8"), ("best P6", "P6"), ("best P7", "P7"), ("consensus", "cons"), ("random 50k #1", "r1"), ("random 50k #2", "r2"), ("60k pick J2384", "pick")):
    co, r = fit(X[k]); print("  %-16s a %+.2f  b %+.2f  c %+.2f  e %+.2f | residual %.3f" % (nme, co[0], co[1], co[2], co[3], r))
base = [(lab == c).mean() for c in (2, 0, 1)]; print("\nlabel base rates: P8 %.3f  P6 %.3f  P7 %.3f" % tuple(base))
out = []
for r in rows:
    gp = G.get(r["key"])
    if gp is None or not gp.exists(): continue
    l = lab[np.load(gp)]; x = nm(lowpass(rd(r["key"]), apix, 10)[envp].astype(np.float64)); co, res = fit(x)
    out.append(dict(key=r["key"], fam=r["fam"], centre=r["centre"], n=r["n"], a=co[0], b=co[1], c=co[2], res=res, f8=float((l == 2).mean()), f6=float((l == 0).mean()), f7=float((l == 1).mean()), P7=r.get("P7")))
print(f"\n{len(out)} window maps with label composition")
A_ = np.array([[1.0, o["f8"], o["f6"], o["f7"]] for o in out]); print("regression of the fitted contents on the label fractions (baseline = other)")
for nme in ("a", "b", "c"):
    y = np.array([o[nme] for o in out]); co = np.linalg.lstsq(A_, y, rcond=None)[0]; r2 = 1 - ((y - A_ @ co) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    bs = []; rng = np.random.default_rng(3)
    for _ in range(400):
        ii = rng.integers(0, len(y), len(y)); bs.append(np.linalg.lstsq(A_[ii], y[ii], rcond=None)[0])
    se = np.std(bs, 0); print(f"  {nme} (content of P{ {'a': 8, 'b': 6, 'c': 7}[nme] }): R^2 {r2:.2f}; per unit label fraction: P8 {co[1]:+.2f}+-{se[1]:.2f}  P6 {co[2]:+.2f}+-{se[2]:.2f}  P7 {co[3]:+.2f}+-{se[3]:.2f}")
print("\nplateau windows (centre 0.375-0.46, n 40-60k):  key  centre  n   | fitted a / b / c | label fractions P8 / P6 / P7 | P7 voxels")
pl = [o for o in out if 0.375 <= o["centre"] <= 0.46 and 40000 <= o["n"] <= 60000]
for o in sorted(pl, key=lambda o: o["centre"]): print("  %-9s %.3f %6d | %+.2f %+.2f %+.2f | %.2f %.2f %.2f | %s" % (o["key"], o["centre"], o["n"], o["a"], o["b"], o["c"], o["f8"], o["f6"], o["f7"], o["P7"]))
c = np.array([o["c"] for o in pl]); f7 = np.array([o["f7"] for o in pl]); print(f"\nplateau windows: P7 content c {c.mean():.2f} +- {c.std():.2f} (range {c.min():.2f}-{c.max():.2f}); P7 label fraction {f7.mean():.2f} +- {f7.std():.2f}; best P7 map has c = 1 by construction, the consensus c = {fit(X['cons'])[0][2]:+.2f}")
json.dump(out, open(H / "p7_unmix.json", "w"), indent=1)
