"""Report for the RESIDUAL-AXIS SPLIT stress test (P17 chains: cryosparc_jobs_kcP17_d1_residseed.json = ab initio seed repeats of J2375 / J2378; d2_resid2 = set_win_rs2: other starting windows, split-aga
Usage: report_residual_split.py -> rs_report.json / rs_report_1120.log"""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np, mrcfile
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from mapdiff import envelope, lowpass
H = Path("/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn/J4001_gP25_WT/method_rank"); W = H.parent; G = W / "candidate_groups_epoch50"; AS = G / "axis_survey"; V7 = G / "presentation_v7"; F = H / "fits"; AL = H / "aligned"
SC = str(Path(__file__).resolve().parents[1] / "tools" / "maps"); CX = "/data/hCFTR/mae2183/tools/chimerax/usr/bin/chimerax"; P = "/home/mae2183/miniconda3/pkgs"
ENV = dict(os.environ, LD_LIBRARY_PATH=f"{P}/mesalib-24.3.4-h0b126fc_1/lib:{P}/libllvm19-19.1.7-hf7376ad_2/lib:/home/mae2183/miniconda3/lib", COREPDB=str(V7 / "models/D1557_fitT.pdb"))
vox = {g["job"]: g for v in json.load(open(H / "method_maps_p17.json"))["methods"].values() for g in v["groups"]}; rows = []
for tag, setdir in (("d1_residseed", None), ("d2_resid2", "set_win_rs2"), ("d3_residax", "set_win_rs3"), ("d4_residrl", "set_win_rs4"), ("d5_residpost", "set_win_rs5"), ("d6_residside", "set_win_rs6")):
    f = AS / f"cryosparc_jobs_kcP17_{tag}.json"
    if not f.exists(): continue
    wins = {w["j"]: w for w in json.load(open(AS / setdir / "windows.json"))["windows"]} if setdir else {}
    for k, v in json.load(open(f)).items():
        j = int(v["group"].split("_g")[-1])
        if setdir: w = wins[j]; rows.append(dict(key="P17_" + v["nu"], set=tag, label=w["label"], n=w["n"], like=0 if "like0" in w["label"] else 1 if "like1" in w["label"] else None))
        else: rows.append(dict(key="P17_" + v["nu"], set=tag, label=f"{'J2375' if j == 0 else 'J2378'} particles, seed {v.get('abinit_seed')}", n=v["n"], like=j))
have = [r for r in rows if (F / f"fit_{r['key']}.mrc").exists()]; print(f"{len(rows)} maps expected, {len(have)} fitted", flush=True)
mask = mrcfile.read(V7 / "models/protein_maskT.mrc").astype(np.float32); apix = float(mrcfile.open(V7 / "models/ref_true.mrc").voxel_size.x); cons = F / "in_J1049.mrc"; todo = []
for r in have:
    for h in "AB":
        name = f"{r['key']}_h{h}"; f = AL / f"{name}.mrc"
        if (F / f"fit_{name}.mrc").exists() or not f.exists(): continue
        if not (F / f"in_{name}.mrc").exists():
            with mrcfile.new(F / f"in_{name}.mrc", overwrite=True) as hh: hh.set_data(mrcfile.read(f)[::-1].astype(np.float32) * mask); hh.voxel_size = apix
        todo.append(f"{F}/fit_{name}.mrc={F}/in_{name}.mrc,{cons}")
for s in range(0, len(todo), 20): subprocess.run([CX, "--nogui", "--offscreen", "--exit", "--script", f"{SC}/cx_pairfit.py"], env=dict(ENV, PAIRS=";".join(todo[s:s + 20])), capture_output=True, text=True)
rd = lambda n: mrcfile.read(F / (f"in_{n}.mrc" if n == "J1049" else f"fit_{n}.mrc")).astype(np.float32); LP = lambda n: lowpass(rd(n), apix, 10); Pm = {k: LP(k) for k in ("J2384", "J2307", "J2334", "J1049")}
envp = envelope(list(Pm.values()), apix, None, k=1.0, dil=3) > 0.5; mean3 = np.mean([Pm[k][envp] for k in ("J2384", "J2307", "J2334")], 0)
def X(n, ab=None):
    v = LP(n)[envp].astype(np.float64); a, b = np.polyfit(v, mean3, 1) if ab is None else ab; return a * v + b, (a, b)
e8, _ = X("J2307"); e6, _ = X("J2334"); u = e6 - e8; un = u @ u; rbar = 0.5 * (X("J1704")[0] + X("J1707")[0]); B = np.stack([X("J2975")[0], X("J2973")[0], X("J2974")[0], np.ones(len(u))], 1); O = {0: X("J2375")[0], 1: X("J2378")[0]}
def score(j):
    x, ab = X(j); co = np.linalg.lstsq(B, x, rcond=None)[0]; r = dict(t=float((x - e8) @ u / un), a=float(co[0]), b=float(co[1]), c=float(co[2]), D_own=None, rho=None, d0=float((x - O[0]) @ (x - O[0]) / un), d1=float((x - O[1]) @ (x - O[1]) / un))
    if all((F / f"fit_{j}_h{h}.mrc").exists() for h in "AB"):
        dA, dB = X(f"{j}_hA", ab)[0] - rbar, X(f"{j}_hB", ab)[0] - rbar; sA, sB = dA @ u / un, dB @ u / un; r.update(D_own=float(dA @ dB / un - sA * sB), rho=float(dA @ dB / np.sqrt((dA @ dA) * (dB @ dB))))
    return r
fm = lambda v, f="%+.3f": (f % v) if v is not None else "  -  "; out = []
print("reference:  PC1 windows c 0.55-0.59, D_own 0.040-0.054 | null D_own 0.026-0.033 | best P7 c 1.00, D_own 0.208")
for j, nm_ in (("J2375", "original P7-like half"), ("J2378", "original P6-like half")):
    r = score(j); print("  %-9s %-36s          | t %5.2f D_own %s rho %s | content P8 %+.2f P6 %+.2f P7 %+.2f" % (j, nm_, r["t"], fm(r["D_own"]), fm(r["rho"], "%+.2f"), r["a"], r["b"], r["c"]))
last = None
for r in have:
    if r["set"] != last: print(f"\n{r['set']}:  key  label  n | LABEL-FREE t, D_own, rho | EVAL voxels P8 / P7 / P6 | content P8 P6 P7 | distance to J2375 / J2378"); last = r["set"]
    s = score(r["key"]); v = vox.get(r["key"], {}); s.update(r, **{c_: v.get(c_, {}).get("vox") for c_ in ("P8", "P7", "P6")}); out.append(s)
    print("  %-9s %-36s %6d | %5.2f %s %s | %6s %6s %6s | %+.2f %+.2f %+.2f | %.3f / %.3f" % (r["key"], r["label"], r["n"], s["t"], fm(s["D_own"]), fm(s["rho"], "%+.2f"), s["P8"], s["P7"], s["P6"], s["a"], s["b"], s["c"], s["d0"], s["d1"]))
for like, nm_ in ((0, "P7-like side"), (1, "P6-like side")):
    for st_ in ("d1_residseed", "d2_resid2", "d3_residax", "d4_residrl", "d5_residpost", "d6_residside"):
        g = [o for o in out if o["set"] == st_ and o["like"] == like and o["D_own"] is not None]
        if g: print(f"\n  {nm_}, {st_} ({len(g)} maps): P7 content {np.mean([o['c'] for o in g]):.2f} +- {np.std([o['c'] for o in g]):.2f} (range {min(o['c'] for o in g):.2f}-{max(o['c'] for o in g):.2f}); D_own {np.mean([o['D_own'] for o in g]):.3f} +- {np.std([o['D_own'] for o in g]):.3f}; P7 voxels {sorted(o['P7'] for o in g)}", end="")
print(); json.dump(out, open(H / "rs_report.json", "w"), indent=1)
