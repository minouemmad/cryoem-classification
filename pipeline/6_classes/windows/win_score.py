"""(11-25) Score the axis windows (label-free picks; heterorefinement maps only to evaluate).  Maps on one scale as od_compare: 10 A, protein mask,
linear fit to the consensus, d2 = |x - y|^2 / |most distant reference pair|^2.
stage 1 (position, sd 0.10, centres 0.1..0.9): end maps = the two extreme windows; pick = interior window farthest from the straight line between them
        (off-line^2 / |end - end|^2) -> prints "STAGE2 <DS> <RUN> <j> <centre>" when no stage-2 set exists yet (the Argon loop builds + queues it).
stage 2 (size at that centre, sd 0.04 .. 0.22 plus the stage-1 0.10 map): pick = the size whose map changes least from its neighbouring sizes
        (mean d2 to the next smaller and next larger size; interior sizes only).  Writes od/win_score.json."""
import glob, json, os, sys, numpy as np, mrcfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../lib" if "pipeline" in __file__ else "../lib")); from fastbp import lowpass
RC = os.environ.get("RES", "/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn") + "/"; MR = RC + "J4001_gP25_WT/method_rank/fits/"; OD = RC + "model_pipeline/od/"; ds = "J4001"
mask = mrcfile.read(RC + "J4001_gP25_WT/candidate_groups_epoch50/presentation_v7/models/protein_maskT.mrc") > 0.5; ap = float(mrcfile.open(MR + "in_J1049.mrc").voxel_size.x)
L = lambda f: lowpass(mrcfile.read(f).astype(np.float32), ap, 10.0)[mask].astype(np.float64); c0 = L(MR + "in_J1049.mrc")
def fit(v): a, b = np.polyfit(v, c0, 1); return a * v + b
ref = {k: fit(L(MR + f)) for k, f in (("P8", "fit_J2975.mrc"), ("P7", "fit_J2974.mrc"), ("P6", "fit_J2973.mrc"), ("Alt0", "fit_Alt0.mrc"), ("Alt1", "fit_Alt1.mrc"))}
un = max(((ref[a] - ref[b]) ** 2).sum() for a in ref for b in ref); d2 = lambda x, y: float(((x - y) ** 2).sum() / un)
rows = {r["name"]: r for r in json.load(open(OD + "cmp/compare_J4001.json"))["rows"]}; cache = {}
def mp(name):
    if name not in cache: cache[name] = fit(L(rows[name]["path"])) if name in rows else None
    return cache[name]
ev = lambda x: {k: round(d2(x, v), 3) for k, v in ref.items()} | {"cons": round(d2(x, c0), 3)}
out = {"consensus_d2": ev(c0)}; print("consensus d2 to refs", out["consensus_d2"])
for s1 in sorted(glob.glob(OD + "J4001/win_*_S?_K9/set.json")):
    sj = json.load(open(s1)); run = sj["run"]; j = int(run[-1]); M = [mp(f"{run} K9: g{k}") for k in range(9)]
    if any(m is None for m in M): print(run, "stage 1 incomplete:", sum(m is not None for m in M), "of 9"); continue
    e0, e1 = M[0], M[8]; u = e1 - e0; L2 = (u ** 2).sum()
    off = [float((((m - e0) - ((m - e0) @ u) / L2 * u) ** 2).sum() / L2) for m in M]; pick = 1 + int(np.argmax(off[1:8])); c = sj["windows"][pick]["centre"]
    out[run] = dict(off_line=[round(v, 4) for v in off], pick=pick, centre=c, ends=[ev(e0), ev(e1)], windows={f"c{w['centre']:.1f}": ev(m) for w, m in zip(sj["windows"], M)})
    best7 = min(range(9), key=lambda k: d2(M[k], ref["P7"]))
    print(f"{run}: off-line^2 {' '.join(f'{v:.3f}' for v in off)} -> pick centre {c} (d2 P7 {d2(M[pick], ref['P7']):.3f}, P8 {d2(M[pick], ref['P8']):.3f}, P6 {d2(M[pick], ref['P6']):.3f})"
          f" | ends nearest: {min(ev(e0), key=ev(e0).get)} / {min(ev(e1), key=ev(e1).get)} | window closest to P7 (eval) centre {sj['windows'][best7]['centre']} d2 {d2(M[best7], ref['P7']):.3f}")
    s2 = OD + f"J4001/{run}_c{int(round(c * 100)):02d}_K6/set.json"
    if not os.path.exists(s2): print(f"STAGE2 {ds} {sj['axis'].split()[0]} {j} {c}"); continue
    sj2 = json.load(open(s2)); r2 = sj2["run"]; M2 = [mp(f"{r2} K6: g{k}") for k in range(6)]
    if any(m is None for m in M2): print(f"  {r2} stage 2 incomplete: {sum(m is not None for m in M2)} of 6"); continue
    sds = [w["sd"] for w in sj2["windows"]] + [0.10]; maps = M2 + [M[pick]]; ns = sj2["sizes"] + [sj["sizes"][pick]]; o = np.argsort(sds); sds = [sds[i] for i in o]; maps = [maps[i] for i in o]; ns = [ns[i] for i in o]
    nb = [d2(maps[i], maps[i + 1]) for i in range(len(maps) - 1)]; stab = [(nb[i - 1] + nb[i]) / 2 for i in range(1, len(maps) - 1)]; ps = 1 + int(np.argmin(stab))
    out[run]["stage2"] = dict(sd=sds, n=ns, neighbour_d2=[round(v, 4) for v in nb], pick_sd=sds[ps], pick_n=ns[ps], pick_eval=ev(maps[ps]), all_eval=[ev(m) for m in maps])
    print(f"  {r2}: sizes {ns} neighbour d2 {' '.join(f'{v:.3f}' for v in nb)} -> pick sd {sds[ps]} ({ns[ps]:,} ptcl) d2 P7 {d2(maps[ps], ref['P7']):.3f} | d2 P7 by size {' '.join(f'{d2(m, ref[chr(80)+chr(55)]):.3f}' for m in maps)}")
json.dump(out, open(OD + "win_score.json", "w"), indent=1)
