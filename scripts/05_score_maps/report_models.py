"""Report for the residual-axis split on OTHER latent-space models / settings (J4001): every cryosparc_jobs_kcP17_f_rv_<TAG>.json = two chains (g0 = lower PC1 rank half, g1 = higher rank half) made by re
Usage: report_models.py -> rv_report.json / rv_report_1122.log"""
import glob, json, os, subprocess, sys
from pathlib import Path
import numpy as np, mrcfile
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from mapdiff import envelope, lowpass
H = Path("/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn/J4001_gP25_WT/method_rank"); W = H.parent; G = W / "candidate_groups_epoch50"; AS = G / "axis_survey"; V7 = G / "presentation_v7"; F = H / "fits"; AL = H / "aligned"
SC = str(Path(__file__).resolve().parents[1] / "tools" / "maps"); CX = "/data/hCFTR/mae2183/tools/chimerax/usr/bin/chimerax"; P = "/home/mae2183/miniconda3/pkgs"
ENV = dict(os.environ, LD_LIBRARY_PATH=f"{P}/mesalib-24.3.4-h0b126fc_1/lib:{P}/libllvm19-19.1.7-hf7376ad_2/lib:/home/mae2183/miniconda3/lib", COREPDB=str(V7 / "models/D1557_fitT.pdb"))
vox = {g["job"]: g for v in json.load(open(H / "method_maps_p17.json"))["methods"].values() for g in v["groups"]}; rows = []
for f in sorted(AS.glob("cryosparc_jobs_kcP17_f_rv_*.json")):
    tag = f.stem.replace("cryosparc_jobs_kcP17_f_rv_", ""); rv = json.load(open(AS / f"set_rv_{tag}/rv.json"))
    for k, v in json.load(open(f)).items():
        g = int(v["group"].split("_g")[-1]); rows.append(dict(key="P17_" + v["nu"], tag=tag, g=g, n=v["n"], make=rv["eval"][f"g{g}"], sep=rv["sep_sd"], ov=rv["overlap_with_original"][g]))
have = [r for r in rows if (F / f"fit_{r['key']}.mrc").exists()]; print(f"{len(rows)} maps expected, {len(have)} fitted", flush=True)
mask = mrcfile.read(V7 / "models/protein_maskT.mrc").astype(np.float32); apix = float(mrcfile.open(V7 / "models/ref_true.mrc").voxel_size.x); cons = F / "in_J1049.mrc"; todo = []
for r in have:
    for h in "AB":
        name = f"{r['key']}_h{h}"; f = AL / f"{name}.mrc"
        if (F / f"fit_{name}.mrc").exists() or not f.exists(): continue
        if not (F / f"in_{name}.mrc").exists():
            with mrcfile.new(F / f"in_{name}.mrc", overwrite=True) as hh: hh.set_data(mrcfile.read(f)[::-1].astype(np.float32) * mask); hh.voxel_size = apix
        todo.append(f"{F}/fit_{name}.mrc={F}/in_{name}.mrc,{cons}")
for s in range(0, len(todo), 20): subprocess.run([CX, "--nogui", "--offscreen", "--exit", "--script", f"{SC}/cx_pairfit.py"], env=dict(ENV, PAIRS=";".join(todo[s:s + 20])), capture_output=True, text=True); print(f"half-map pairfit {min(s + 20, len(todo))}/{len(todo)}", flush=True)
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
print("reference:  PC1 windows c 0.55-0.59, D_own 0.040-0.054 | null D_own 0.026-0.033 | base residual split J2375 c 0.69 D_own 0.107 | best P7 c 1.00, D_own 0.208")
for j, nm_ in (("J2375", "base split, P7-like half"), ("J2378", "base split, P6-like half"), ("J2974", "best P7")):
    r = score(j); print("  %-9s %-28s | t %5.2f D_own %s rho %s | content P8 %+.2f P6 %+.2f P7 %+.2f" % (j, nm_, r["t"], fm(r["D_own"]), fm(r["rho"], "%+.2f"), r["a"], r["b"], r["c"])); out.append(dict(tag="base", key=j, g=None, **r))
last = None
for r in have:
    if r["tag"] != last: print(f"\n{r['tag']}: sep {r['sep']:.2f} sd"); last = r["tag"]
    s = score(r["key"]); v = vox.get(r["key"], {}); s.update(r, **{c_: v.get(c_, {}).get("vox") for c_ in ("P8", "P7", "P6")}); out.append(s)
    print("  g%d %-9s %6d | t %5.2f D_own %s rho %s | vox P8 %5s P7 %5s P6 %5s | content P8 %+.2f P6 %+.2f P7 %+.2f | d(J2375) %.3f d(J2378) %.3f | share in orig halves %.2f / %.2f | %s" % (r["g"], r["key"], r["n"], s["t"], fm(s["D_own"]), fm(s["rho"], "%+.2f"), s["P8"], s["P7"], s["P6"], s["a"], s["b"], s["c"], s["d0"], s["d1"], r["ov"][0], r["ov"][1], r["make"]))
json.dump(out, open(H / "rv_report.json", "w"), indent=1)
