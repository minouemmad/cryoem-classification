"""Align, fit and voxel-score the maps of the cryoSPARC chains in project P17."""
import glob, json, os, subprocess, sys
from pathlib import Path
import numpy as np, mrcfile
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from mapdiff import diff, envelope, lowpass
H = Path(os.environ.get("MR_DIR", "/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn/J4001_gP25_WT/method_rank")); W = H.parent; G = W / "candidate_groups_epoch50"; AS = G / "axis_survey"; V7 = G / "presentation_v7"
F = H / "fits"; AL = H / "aligned"
SC = str(Path(__file__).resolve().parents[1] / "tools" / "maps"); CX = "/data/hCFTR/mae2183/tools/chimerax/usr/bin/chimerax"; P = "/home/mae2183/miniconda3/pkgs"
ENV = dict(os.environ, LD_LIBRARY_PATH=f"{P}/mesalib-24.3.4-h0b126fc_1/lib:{P}/libllvm19-19.1.7-hf7376ad_2/lib:/home/mae2183/miniconda3/lib", COREPDB=str(V7 / "models/D1557_fitT.pdb"))
P17D = "/data/hCFTR/mae2183/cs_projects/CS-j4001-rae-window-centre-scan-plateau-sizes"
REFS = {"consensus": "J1049", "best_P6": "J2973", "best_P7": "J2974", "best_P8": "J2975", "rnd50k_g0": "J1704", "rnd50k_g1": "J1707"}
METHODS = {}
for f in sorted(AS.glob("cryosparc_jobs_kcP17_*.json")):
    if "j4624" in f.name: continue                                   # J4624 chains are scored separately (other dataset)
    tag = f.stem.replace("cryosparc_jobs_", "")
    for k, v in json.load(open(f)).items():
        if not isinstance(v, dict) or not v.get("nu"):
            continue
        g = v.get("group", k).split("/")[-1]
        METHODS.setdefault(f"{tag}:{g}", []).append(dict(job="P17_" + v["nu"], n=v.get("n"), group=g, centre=None))
idx = {}
RAEA = AS / "J4001_rae_nuposes/compare/aligned"
for k, j in REFS.items():
    idx[j] = RAEA / f"{k}.mrc"
need = sorted({g["job"] for gs in METHODS.values() for g in gs} | set(REFS.values()))
jd = lambda j: (P17D, j[4:])
def done(j):
    try:
        return j[4:] in set(json.load(open(H / "p17_completed.json")))
    except Exception:
        return False
miss = [j for j in need if j.startswith("P17_") and not (AL / f"{j}.mrc").exists() and done(j) and glob.glob(f"{jd(j)[0]}/{jd(j)[1]}/{jd(j)[1]}_*_volume_map.mrc")]
print(f"{len(METHODS)} chains, {len(need)} maps, {len(miss)} to align", flush=True)
if miss:
    json.dump(dict(out=str(AL), maps={j: list(jd(j)) for j in miss}), open(H / "align_spec_p17.json", "w"), indent=1)
    subprocess.run([CX, "--nogui", "--offscreen", "--exit", "--script", str(Path(__file__).resolve().parent / "align_nu_maps_chimerax.py")], env=dict(ENV, SPEC=str(H / "align_spec_p17.json")), capture_output=True, text=True)
for j in need:
    if j.startswith("P17_") and (AL / f"{j}.mrc").exists():
        idx[j] = AL / f"{j}.mrc"
mask = mrcfile.read(V7 / "models/protein_maskT.mrc").astype(np.float32); apix = float(mrcfile.open(V7 / "models/ref_true.mrc").voxel_size.x)
for j in need:
    if j in idx and not (F / f"in_{j}.mrc").exists():
        v = mrcfile.read(idx[j])
        v = v[::-1].astype(np.float32) * mask
        with mrcfile.new(F / f"in_{j}.mrc", overwrite=True) as h:
            h.set_data(v); h.voxel_size = apix
cons = F / f"in_{REFS['consensus']}.mrc"
todo = [f"{F}/fit_{j}.mrc={F}/in_{j}.mrc,{cons}" for j in need if j != REFS["consensus"] and (F / f"in_{j}.mrc").exists() and not (F / f"fit_{j}.mrc").exists()]
for s in range(0, len(todo), 20):
    subprocess.run([CX, "--nogui", "--offscreen", "--exit", "--script", f"{SC}/cx_pairfit.py"], env=dict(ENV, PAIRS=";".join(todo[s:s + 20])), capture_output=True, text=True)
    print(f"pairfit {min(s + 20, len(todo))}/{len(todo)}", flush=True)
M = lambda j: mrcfile.read(F / (f"in_{j}.mrc" if j == REFS["consensus"] else f"fit_{j}.mrc")).astype(np.float32)
R = {k: M(j) for k, j in REFS.items()}
env = envelope(list(R.values()), apix, 10, k=1.0, dil=3); ev = env > 0.5
thr = 1.02 * float(np.abs(diff(R["rnd50k_g1"], R["rnd50k_g0"], apix, 10, env)[0][ev]).max())
LB = {c: lowpass(R[f"best_{c}"], apix, 10)[ev] for c in ("P6", "P7", "P8")}


def score(v):
    out = {}
    for c in ("P6", "P7", "P8"):
        d, _, _ = diff(v, R[f"best_{c}"], apix, 10, env); out[c] = dict(vox=int((np.abs(d[ev]) > thr).sum()), cc=float(np.corrcoef(lowpass(v, apix, 10)[ev], LB[c])[0, 1]))
    return out


print("threshold", thr, flush=True)
res = {}
for name, gs in METHODS.items():
    rows = []
    for g in gs:
        if not (F / f"fit_{g['job']}.mrc").exists():
            continue
        rows.append(dict(g, **score(M(g["job"]))))
    if rows:
        res[name] = dict(groups=rows)
        print(name, {c: (rows[0][c]["vox"]) for c in ("P7", "P6", "P8")}, rows[0]["n"], flush=True)
json.dump(dict(threshold=thr, methods=res), open(H / "method_maps_p17.json", "w"), indent=1)
