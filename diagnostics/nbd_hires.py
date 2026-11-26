"""(11-26) NBD1 detail check: per-domain band power (10-6 A) relative to the same domain in the P8 reference, and per-domain corr with P8 at 6 A, for the P8-like maps (EVALUATION ONLY)."""
import os
import json, sys, numpy as np, mrcfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../lib" if "pipeline" in __file__ else "../lib")); from mapdiff import lowpass
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "nbd_density.py")).read().split("L = lambda")[0])     # reuse masks (lab, DOM, ap, REF, cons, d)
o = json.load(open(R + "model_pipeline/od/nbd_density.json")); p8 = sorted([r for r in o["rows"] if r["closest"] == "P8"], key=lambda r: r["d2"]["P8"])[:8]
path = {r["name"]: r["path"] for r in d["rows"]}
def bands(f):
    v = mrcfile.read(f).astype(np.float32); l6 = lowpass(v, ap, 6.0); l10 = lowpass(v, ap, 10.0); hb = l6 - l10; return l6, l10, hb
P6, P10, PH = bands(REF["P8"]); C6, C10, CH = bands(cons)
tm = (lab == 1) | (lab == 3)
def score(f):
    l6, l10, hb = bands(f); a, b = np.polyfit(l10[tm], P10[tm], 1)   # scale on TMDs to the P8 ref
    l6, hb = a * l6 + b, a * hb
    s = {}
    for n, dn in enumerate(DOM, 1):
        m = lab == n; s[dn] = (float(np.sqrt((hb[m] ** 2).mean() / (PH[m] ** 2).mean())), float(np.corrcoef(l6[m], P6[m])[0, 1]))
    return s
print("columns: domain hi-band(10-6A) amplitude vs P8 ref / corr with P8 ref at 6 A")
for nm, f in [("REF P7", REF["P7"]), ("REF P6", REF["P6"]), ("consensus", cons)] + [(r["name"] + f" (n {r['n']})", path[r["name"]]) for r in p8]:
    s = score(f); print(f"{nm:34s}", "  ".join(f"{dn} {a:.2f}/{c:.2f}" for dn, (a, c) in s.items()), flush=True)
