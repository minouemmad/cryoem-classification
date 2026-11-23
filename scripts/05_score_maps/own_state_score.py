"""Label-free own-state score D_own: signal shared by the two half maps of a window, relative to random-subset reference maps."""
import json, sys
from pathlib import Path
import numpy as np, mrcfile
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from mapdiff import envelope, lowpass
H = Path("/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn/J4001_gP25_WT/method_rank"); F = H / "fits"
rows = json.load(open(H / "expl_report.json")); U = {o["key"]: o for o in json.load(open(H / "p7_unmix.json"))}; byk = {}
for r in rows: byk.setdefault(r["key"], r)
apix = float(mrcfile.open(F / "fit_J2307.mrc").voxel_size.x); rd = lambda n: mrcfile.read(F / (f"in_{n}.mrc" if n == "J1049" else f"fit_{n}.mrc")).astype(np.float32); out = {}
for RES in (10, 8):
    LP = lambda n: lowpass(rd(n), apix, RES); Pm = {k: LP(k) for k in ("J2384", "J2307", "J2334", "J1049")}
    envp = envelope(list(Pm.values()), apix, None, k=1.0, dil=3) > 0.5; mean3 = np.mean([Pm[k][envp] for k in ("J2384", "J2307", "J2334")], 0)
    def X(n, ab=None):
        v = LP(n)[envp].astype(np.float64); a, b = np.polyfit(v, mean3, 1) if ab is None else ab; return a * v + b, (a, b)
    e8, _ = X("J2307"); e6, _ = X("J2334"); u = e6 - e8; un = u @ u; r1, _ = X("J1704"); r2, _ = X("J1707"); rbar = 0.5 * (r1 + r2)
    def sc(j, ref):
        x, ab = X(j); dA, dB = X(f"{j}_hA", ab)[0] - ref, X(f"{j}_hB", ab)[0] - ref; sA, sB = dA @ u / un, dB @ u / un
        return dict(t=float((x - e8) @ u / un), D_rnd=float(dA @ dB / un), D_own=float(dA @ dB / un - sA * sB), rho=float(dA @ dB / np.sqrt((dA @ dA) * (dB @ dB))))
    print(f"\n################ {RES} A   reference = random 50k subset maps   (units of |e6 - e8|^2)")
    null = [sc("J1704", r2), sc("J1707", r1)]; print("  NULL  random subset vs the other random subset:  D_rnd %+.4f / %+.4f   D_own %+.4f / %+.4f   rho %+.2f / %+.2f" % (null[0]["D_rnd"], null[1]["D_rnd"], null[0]["D_own"], null[1]["D_own"], null[0]["rho"], null[1]["rho"]))
    for j, nm_ in (("J2974", "best P7"), ("J2973", "best P6"), ("J2975", "best P8"), ("J2307", "pipeline P8 end"), ("J2334", "pipeline P6 end"), ("J2384", "60k pick")):
        r = sc(j, rbar); out[f"{RES}:{j}"] = r; print("  %-9s %-16s t %5.2f  D_rnd %+.4f  D_own %+.4f  rho %+.2f" % (j, nm_, r["t"], r["D_rnd"], r["D_own"], r["rho"]))
    tab = []
    for k, r0 in byk.items():
        if r0["fam"] in ("ktilt", "kalt"): continue
        r = sc(k, rbar); r.update(key=k, fam=r0["fam"], centre=r0["centre"], n=r0["n"], P7=r0.get("P7"), c=U.get(k, {}).get("c")); out[f"{RES}:{k}"] = r; tab.append(r)
    for size, lo, hi in (("43k", 40000, 47000), ("58k", 54000, 62000)):
        g = sorted([r for r in tab if lo <= r["n"] <= hi and r["fam"] not in ("kknn", "krep")], key=lambda r: r["centre"])
        print(f"  by centre, {size}:  centre: D_own (rho) [EVAL c]"); print("    " + "  ".join(f"{r['centre']:.3g}: {r['D_own']:+.3f} ({r['rho']:+.2f}) [{r['c']:.2f}]" for r in g if r["c"] is not None))
    for nm_, sel in (("replicates 43k @0.445", [r for r in tab if r["fam"] == "krep" and r["n"] < 50000]), ("replicates 58k @0.445", [r for r in tab if r["fam"] == "krep" and r["n"] >= 50000])):
        print(f"  {nm_}: D_own {np.mean([r['D_own'] for r in sel]):+.4f} +- {np.std([r['D_own'] for r in sel]):.4f} (rho {np.mean([r['rho'] for r in sel]):+.2f});  null {null[0]['D_own']:+.4f} / {null[1]['D_own']:+.4f}")
    g = sorted([r for r in tab if abs(r["centre"] - 0.4302) < 1e-3], key=lambda r: r["n"]); print("  by size at centre 0.43:  " + "  ".join(f"{r['n'] // 1000}k: {r['D_own']:+.3f}" for r in g))
    v = [r for r in tab if r["c"] is not None]; cc = np.array([r["c"] for r in v]); big = [r for r in v if r["n"] >= 40000]
    print(f"  EVAL: correlation of P7 content c with D_own over {len(v)} windows {np.corrcoef(cc, [r['D_own'] for r in v])[0, 1]:+.2f}; over the {len(big)} windows of >= 40k {np.corrcoef([r['c'] for r in big], [r['D_own'] for r in big])[0, 1]:+.2f}")
json.dump(out, open(H / "consensus_score2.json", "w"), indent=1)
