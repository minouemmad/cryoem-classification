"""Label-free rule for placing the P7 window: departure of each PC1-window map (own ab initio + NU, method_rank frame) from the straight line between the pipeline's OWN two end-window maps.
Usage: offline_rule.py   -> method_rank/offline_rule.json"""
import json, sys
from pathlib import Path
import numpy as np, mrcfile
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from mapdiff import envelope, lowpass
H = Path("/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn/J4001_gP25_WT/method_rank"); F = H / "fits"; AS = H.parent / "candidate_groups_epoch50/axis_survey"
MM = json.load(open(H / "method_maps.json"))["methods"]; vox = {g["job"]: g for v in MM.values() for g in v["groups"]}
apix = float(mrcfile.open(F / "fit_J2307.mrc").voxel_size.x); L = lambda j: lowpass(mrcfile.read(F / f"fit_{j}.mrc").astype(np.float32), apix, 10)
out = {}
def run(tag, ends, items):
    """ends: (P8-end job, P6-end job); items: list of (label, job)"""
    maps = {j: L(j) for j in {*ends, *[j for _, j in items]}}
    ev = envelope([maps[j] for j in ends] + [maps[j] for _, j in items], apix, None, k=1.0, dil=3) > 0.5
    mean = np.mean([m[ev] for m in maps.values()], 0)
    def norm(v):
        a, c = np.polyfit(v, mean, 1); return a * v + c
    X = {j: norm(m[ev].astype(np.float64)) for j, m in maps.items()}; e8, e6 = X[ends[0]], X[ends[1]]; u = e6 - e8; un = u @ u
    print(f"\n=== {tag}: ends {ends}, envelope voxels {int(ev.sum())}"); print(f"{'window':34s} {'job':6s}   t     off-line^2   | eval: P7 vox  P6 vox  P8 vox"); rows = []
    for lab, j in items:
        t = float((X[j] - e8) @ u / un); r = X[j] - e8 - t * u; off = float(r @ r / un); g = vox.get(j, {})
        rows.append(dict(label=lab, job=j, t=t, off2=off, P7=g.get("P7", {}).get("vox"), P6=g.get("P6", {}).get("vox"), P8=g.get("P8", {}).get("vox")))
        print(f"{lab:34s} {j:6s} {t:6.3f}   {off:8.4f}     |       {rows[-1]['P7']!s:>6} {rows[-1]['P6']!s:>7} {rows[-1]['P8']!s:>7}")
    b = max(rows, key=lambda r: r["off2"]); print(f"-> rule picks: {b['label']} ({b['job']}), off-line^2 {b['off2']:.4f}; fewest P7 voxels: {min((r for r in rows if r['P7'] is not None), key=lambda r: r['P7'])['label']}")
    out[tag] = rows
for s in ("rae", "vae"):
    st = json.load(open(AS / f"cryosparc_jobs_set_win_{s}.json")); w = json.load(open(AS / f"set_win_{s}/windows.json"))["windows"]
    it = [(f"window {x['j']} rank {x['rank_lo']:.2f}-{x['rank_hi']:.2f}", v["nu"]) for x, v in zip(w, st.values())]
    run(f"{s.upper()} sliding 60k windows", (it[0][1], it[-1][1]), it)
    if s == "rae" and (AS / "cryosparc_jobs_set_win_fine.json").exists():
        sf = json.load(open(AS / "cryosparc_jobs_set_win_fine.json")); wf = json.load(open(AS / "set_win_fine/windows.json"))["windows"]
        al = sorted([(x["rank_lo"], x["rank_hi"], v["nu"], "coarse") for x, v in zip(w, st.values())] + [(x["rank_lo"], x["rank_hi"], v["nu"], "fine") for x, v in zip(wf, sf.values())])
        al = [a_ for a_ in al if (F / f"fit_{a_[2]}.mrc").exists()]
        run("RAE coarse + fine 60k windows (ends = the coarse end windows)", (it[0][1], it[-1][1]), [(f"rank {lo:.2f}-{hi:.2f} ({kind})", j) for lo, hi, j, kind in al])
    if s == "rae":
        ex = [(f"K4 group {i}", j) for i, j in enumerate(("J1536", "J1539", "J1542", "J1545"))] + [(f"K4 g1 replicate {i + 1}", v["nu"]) for i, v in enumerate(json.load(open(AS / "cryosparc_jobs_set_win_rep.json")).values())]
        ex += [(f"K3 group {i}", j) for i, j in enumerate(("J1527", "J1530", "J1533"))] + [("residual cluster (P7-like half)", "J2375"), ("residual complement (control)", "J2378")]
        run("RAE k-means groups and residual split, against the RAE window ends", (it[0][1], it[-1][1]), ex)
json.dump(out, open(H / "offline_rule.json", "w"), indent=1)
