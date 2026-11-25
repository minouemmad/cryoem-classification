"""Evaluation only: refined label-free class maps (every refined set, run x K) vs the heterorefinement class maps refined by the SAME route.
 1. every finished NU map -> deck frame (cx_align_nu.py: rigid fit to $DECK_REF, both hands) -> true hand ([::-1]) x protein mask;
 2. consensus = particle-weighted mean of the dataset's refined heterorefinement maps; every map refitted on the TMD core onto it (cx_pairfit.py);
 3. at 10 A inside the envelope, after a linear density scale fit x -> a x + b onto the consensus c0:
       cc_k = corr(x - c0, class_k - c0)          d2_k = |x - class_k|^2 / |class_a - class_b|^2   (a, b = most different heterorefinement pair)
 4. per refined set: Hungarian match of its maps to the heterorefinement classes by cc; solid renders of every map.
Incremental (aligns / fits / renders only new maps).  Usage: compare.py DS [DS ...]   -> $WORK/compare/compare_<DS>.json, $WORK/compare/rend/"""
import glob, json, os, re, subprocess, sys
from pathlib import Path
import numpy as np, mrcfile
from scipy.optimize import linear_sum_assignment
H = Path(__file__).resolve().parent; sys.path.insert(0, str(H.parents[1] / "lib"))
from mapdiff import envelope, lowpass
from maps import cc

E = os.environ; W = E["WORK"] + "/"; O = W + "compare/"; PD = E["CS_PROJECT_DIR"]; DECK = E["DECK"] + "/"
CX = lambda script, **env: subprocess.run([E["CHIMERAX"], "--nogui", "--offscreen", "--exit", "--script", str(H / script)], capture_output=True, text=True,
                                          env=dict(E, LD_LIBRARY_PATH=E["CX_LD"], COREPDB=str(H.parents[1] / "reference/pdb/D1557_fitT.pdb"), **env))
slug = lambda t: re.sub(r"[^A-Za-z0-9]+", "_", t).strip("_")
mask = mrcfile.read(DECK + "models/protein_maskT.mrc").astype(np.float32); apx = float(mrcfile.open(DECK + "models/ref_true.mrc").voxel_size.x)
done = lambda j: bool(glob.glob(f"{PD}/{j}/{j}_*_volume_map_half_B.mrc"))
for ds in sys.argv[1:]:
    AL = O + f"{ds}/"; os.makedirs(AL, exist_ok=True); items = []
    for f in sorted(glob.glob(W + f"refine/cs_{ds}_*_K*.json")):
        run, K = re.search(rf"cs_{ds}_(.+)_K(\d+)\.json$", f).groups(); K = int(K); sj = json.load(open(W + f"refine/{ds}/{run}_K{K}/set.json"))
        for k, (nm, v) in enumerate(sorted(json.load(open(f)).items())):
            lab = nm.rsplit("_", 1)[-1]; size = sj["sizes"][int(lab[1:]) if lab.startswith("g") else sj["names"].index(lab)]
            if done(v["nu"]): items.append(dict(ds=ds, run=run, K=K, k=k, job=v["nu"], n=size, name=f"class {lab}" if run == "hetero" else f"{run} K{K}: {lab}"))
    het = [it for it in items if it["run"] == "hetero"]; nh = len(glob.glob(W + f"refine/{ds}/hetero_K*/cryosparc/*.star"))
    if not het or len(het) < nh: print(ds, f"heterorefinement maps not all refined yet ({len(het)} of {nh})"); continue
    # 1. deck frame, hand, mask
    raw = {slug(it["name"]): [PD, it["job"]] for it in items if not os.path.exists(AL + f"in_{slug(it['name'])}.mrc")}
    if raw:
        json.dump(dict(out=AL, maps=raw), open(AL + "spec.json", "w")); CX("cx_align_nu.py", SPEC=AL + "spec.json")
        for n in raw:
            if not os.path.exists(AL + f"{n}.mrc"): print("NOT ALIGNED", ds, n, flush=True); continue
            with mrcfile.new(AL + f"in_{n}.mrc", overwrite=True) as h: h.set_data(mrcfile.read(AL + f"{n}.mrc")[::-1].astype(np.float32) * mask); h.voxel_size = apx
    # 2. consensus and core refit
    cons = AL + "consensus_hetero_mean.mrc"
    if not os.path.exists(cons):
        w = np.array([it["n"] for it in het], float); M = sum(wi * mrcfile.read(AL + f"in_{slug(it['name'])}.mrc").astype(np.float64) for wi, it in zip(w, het)) / w.sum()
        with mrcfile.new(cons, overwrite=True) as h: h.set_data(M.astype(np.float32)); h.voxel_size = apx
    todo = [f"{AL}fit_{slug(it['name'])}.mrc={AL}in_{slug(it['name'])}.mrc,{cons}" for it in items if os.path.exists(AL + f"in_{slug(it['name'])}.mrc") and not os.path.exists(AL + f"fit_{slug(it['name'])}.mrc")]
    for i in range(0, len(todo), 20): CX("cx_pairfit.py", PAIRS=";".join(todo[i:i + 20]))
    items = [it for it in items if os.path.exists(AL + f"fit_{slug(it['name'])}.mrc")]; het = [it for it in items if it["run"] == "hetero"]
    # 3. cc_k and d2_k at 10 A
    L = lambda f: lowpass(mrcfile.read(f).astype(np.float32), apx, 10.0)
    Cm = {it["name"][6:]: L(AL + f"fit_{slug(it['name'])}.mrc") for it in het}; c0f = L(cons)
    env = envelope(list(Cm.values()) + [c0f], apx, None, k=1.0, dil=3) > 0.5; c0 = c0f[env].astype(np.float64)

    def scaled(v):
        v = v[env].astype(np.float64); a, b = np.polyfit(v, c0, 1); return a * v + b
    Cx = {k: scaled(v) for k, v in Cm.items()}; ks = list(Cx)
    pair = max(((a, b) for a in ks for b in ks if a < b), key=lambda p: ((Cx[p[0]] - Cx[p[1]]) ** 2).sum()); un = ((Cx[pair[0]] - Cx[pair[1]]) ** 2).sum(); rows = []
    for it in items:
        x = scaled(L(AL + f"fit_{slug(it['name'])}.mrc"))
        r = dict(it, path=AL + f"fit_{slug(it['name'])}.mrc", cc_diff={k: cc(x - c0, Cx[k] - c0) for k in ks}, d2={k: float(((x - Cx[k]) ** 2).sum() / un) for k in ks}, scale_pair=list(pair))
        r["closest"] = min(r["d2"], key=r["d2"].get); rows.append(r)
    # 4. matching per refined set
    sets = {}
    for run, K in sorted({(r["run"], r["K"]) for r in rows if r["run"] != "hetero"}):
        S = [r for r in rows if r["run"] == run and r["K"] == K]
        if len(S) < K: continue
        C = np.array([[r["cc_diff"][k] for k in ks] for r in S]); i, j = linear_sum_assignment(-C)
        sets[f"{run} K{K}"] = s = dict(run=run, K=K, match={S[a]["name"]: ks[b] for a, b in zip(i, j)}, match_cc=[float(C[a, b]) for a, b in zip(i, j)], mean_match_cc=float(C[i, j].mean()),
                                       covered=sorted({ks[b] for a, b in zip(i, j) if C[a, b] >= 0.5}), best_any=[max(r["cc_diff"].values()) for r in S])
        print(f"  {run} K{K}: matched {s['match']} cc {' '.join(f'{v:.2f}' for v in s['match_cc'])}; classes with a counterpart (cc >= 0.5): {s['covered']}", flush=True)
    json.dump(dict(ds=ds, classes=ks, hetero_cc={a: {b: cc(Cx[a] - c0, Cx[b] - c0) for b in ks} for a in ks}, rows=rows, sets=sets), open(O + f"compare_{ds}.json", "w"), indent=1)
    RD = O + "rend/"; os.makedirs(RD + "maskedT", exist_ok=True); names, solid = [], []
    for r in rows:
        n = slug(ds + "_" + r["name"])
        if not os.path.exists(RD + f"renders/solid_{n}.png"):
            subprocess.run(["cp", r["path"], RD + f"maskedT/{n}.mrc"]); names.append(n); solid.append(f"{n}=" + ("#4A5D7E" if r["run"] == "hetero" else "#2A9D8F"))
    if names:
        json.dump(dict(diff_sd=1.0), open(RD + "maskedT/render.json", "w")); p = CX("cx_render_maps.py", D=RD, NAMES=" ".join(names), SOLID=" ".join(solid))
        print(ds, "rendered", p.stdout.count("RENDER"), "of", len(names), flush=True)
