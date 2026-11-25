"""One STAR per class for cryoSPARC (import without poses -> ab initio -> NU), rewritten to the local 128 px stack (row order of the consensus
STAR = cryoDRGN particle order; optics pixel size scaled by box / 128).
RUN = a pipeline run (classes of pipe/<DS>/<RUN>/mc at each K)  |  'hetero' = the heterorefinement classes (config/heteroref.json), refined by
the same route as the reference for the comparison.
Usage: stars.py DS RUN K [K ...]   -> $WORK/refine/<DS>/<RUN>_K<K>/{labels.npy, set.json, cryosparc/<DS>_<RUN>_K<K>_<g#|C##>.star}"""
import glob, json, os, re, sys
from pathlib import Path
import numpy as np, mrcfile

ROOT = Path(__file__).resolve().parents[2]; RES, WORK = Path(os.environ["RES"]), Path(os.environ["WORK"])
ds, run, Ks = sys.argv[1], sys.argv[2], [int(k) for k in sys.argv[3:]]; base = ds.rstrip("c"); cfg = json.load(open(ROOT / "config/heteroref.json"))[base]; W = RES / cfg["dir"]
if run == "hetero":   # class membership by uid, in stack order
    u = np.load(W / cfg["poses_cs"])["uid"]; pos = {x: i for i, x in enumerate(u)}; files = sorted(glob.glob(str(W / cfg["classes"])))
    hl = -np.ones(len(u), np.int64); NAMES = []
    for k, f in enumerate(files):
        hl[[pos[x] for x in np.load(f)["uid"]]] = k; NAMES.append("C%02d" % int(re.search(r"class_(\d+)", f).group(1)))
    assert (hl >= 0).all(); cell = np.arange(len(u)); z = {f"K{len(files)}": hl}; Ks = [len(files)]
else:
    NAMES = None; mc = WORK / f"pipe/{ds}/{run}/mc"; z = np.load(mc / "MP_model_labels.npz"); cell = np.load(mc / "MP_model_cells.npz")["cell"]
stack = W / "inputs/particles.128.mrcs"
with mrcfile.mmap(stack, permissive=True) as m: n, D = m.data.shape[0], m.data.shape[-1]
star = next(p for p in [W / f"nu_consensus/{cfg['dir']}_particles.star", W / f"nu_consensus/{base}_particles.star"] if p.exists())
full = star.read_text().split("\n"); dp = next(i for i, l in enumerate(full) if l.strip() == "data_particles")
he = next(i for i in range(dp + 1, len(full)) if full[i].strip() and not full[i].startswith("_") and full[i].strip() != "loop_"); rows = [l for l in full[he:] if l.strip()]
assert len(rows) == n == len(cell), (len(rows), n, len(cell)); hdr = []
for l in full[:he]:
    p = l.split()
    if len(p) == 8 and p[7].startswith("opticsGroup"): p[1] = "%.6f" % (float(p[1]) * int(p[5]) / D); p[5] = str(D); l = " ".join(p)
    hdr.append(l)
new = [" ".join(["%06d@%s" % (i + 1, stack)] + r.split()[1:]) for i, r in enumerate(rows)]
for K in Ks:
    lab = z[f"K{K}"][cell]; O = WORK / f"refine/{ds}/{run}_K{K}"; (O / "cryosparc").mkdir(parents=True, exist_ok=True); np.save(O / "labels.npy", lab.astype(np.int16)); out = []
    for k in range(K):
        f = O / "cryosparc" / f"{ds}_{run}_K{K}_{NAMES[k] if NAMES else f'g{k}'}.star"; f.write_text("\n".join(hdr + [r for r, l in zip(new, lab) if l == k]) + "\n\n"); out.append(str(f))
    json.dump(dict(ds=ds, run=run, K=K, names=NAMES, sizes=np.bincount(lab, minlength=K).tolist(), stars=out), open(O / "set.json", "w"), indent=1)
    print(ds, run, "K", K, np.bincount(lab, minlength=K).tolist())
