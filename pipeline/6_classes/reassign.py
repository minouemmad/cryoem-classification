"""Optional: particle-level reassignment to the class maps with FIXED poses (one 3D-classification-like refinement of the label-free classes).
Start: the pipeline's K classes (mc/MP_model_labels.npz, K = elbow unless MP_RE_K).  Each iteration:
  two random folds; K class maps are reconstructed from fold-A particles only and every fold-B particle is scored against each of them (and vice versa),
  so no particle is ever scored against a map that contains it.  Score = <x_i, c_i P_ik> / ||c_i P_ik|| over the band MP_RE_LO-MP_RE_HI A (FastBP.fit;
  scale-free in the map).  Maps are low-passed to the band edge so classes of different size are compared at the same resolution.  Hard argmax.
  Stops when < 1 % of particles change or after MP_RE_IT iterations.
Writes MP_OUT/reassign/: labels_it<t>.npy, summary.json (sizes, changed fraction, ARI vs start, mean score margin), class_k.mrc (full-res 128 px maps)
and halfA/B_k.mrc of the final classes.  Null (MP_RE_NULL=1): same procedure on a simulated homogeneous stack with the real poses/CTF/noise."""
import json, os, sys, time
from pathlib import Path
import numpy as np, torch, mrcfile
from sklearn.metrics import adjusted_rand_score
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
from fastbp import FastBP

MO = os.environ["MP_OUT"] + "/"; NULL = os.environ.get("MP_RE_NULL", "") == "1"; O = MO + ("reassign_null/" if NULL else "reassign/"); os.makedirs(O, exist_ok=True)
LO = float(os.environ.get("MP_RE_LO", "50")); HI = float(os.environ.get("MP_RE_HI", "8")); IT = int(os.environ.get("MP_RE_IT", "8"))
mc = json.load(open(MO + "mc/MP.json")); K = int(os.environ.get("MP_RE_K", "0")) or int(mc["model"]["elbow"])
z = np.load(MO + "mc/MP_model_labels.npz"); lab0 = z[f"K{K}"][z["cell"]].astype(int)
t0 = time.time(); say = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
bp = FastBP(os.environ["MP_STACK"], os.environ["MP_POSES"], os.environ["MP_CTF"]); N = bp.data.N; assert len(lab0) == N, (len(lab0), N)
if NULL: bp.simulate(seed=7)
band = bp.band_mask(LO, HI); say(f"K {K}, start sizes {np.bincount(lab0, minlength=K).tolist()}, band {LO}-{HI} A ({int(band.sum())} px), null {NULL}")
D = bp.D; r3 = None
def lp_h(H):
    """zero the Hartley map beyond the band's high-resolution edge (same resolution for every class map)"""
    global r3
    if r3 is None:
        g = torch.arange(D, device=H.device) - D // 2; r3 = torch.sqrt(g[:, None, None] ** 2 + g[None, :, None] ** 2 + g[None, None, :] ** 2) / (D - 1) / bp.apix
    return H * (r3 <= 1.0 / HI)
lab = lab0.copy(); hist = []
for it in range(1, IT + 1):
    fold = np.random.default_rng(100 + it).random(N) < 0.5; S = np.empty((N, K))
    for f in (0, 1):
        src, dst = np.flatnonzero(fold == bool(f)), np.flatnonzero(fold != bool(f))
        maps = [lp_h(bp.hvol(src[lab[src] == k])) for k in range(K)]
        for k in range(K): S[dst, k] = bp.fit(dst, maps, np.full(len(dst), k), band)
        del maps; torch.cuda.empty_cache()
    new = S.argmax(1); ch = float((new != lab).mean()); srt = np.sort(S, 1); margin = float(np.median(srt[:, -1] - srt[:, -2]) / np.median(np.abs(srt[:, -1])))
    lab = new; np.save(O + f"labels_it{it}.npy", lab.astype(np.int8)); np.save(O + f"scores_it{it}.npy", S.astype(np.float32))
    h = dict(it=it, sizes=np.bincount(lab, minlength=K).tolist(), changed=ch, ari_vs_start=float(adjusted_rand_score(lab0, lab)), median_rel_margin=margin)
    hist.append(h); say(h); json.dump(dict(K=K, band=[LO, HI], null=NULL, start_sizes=np.bincount(lab0, minlength=K).tolist(), iters=hist), open(O + "summary.json", "w"), indent=1)
    if ch < 0.01: break
np.save(O + "labels_final.npy", lab.astype(np.int8))
if not NULL:
    for k in range(K):
        idx = np.flatnonzero(lab == k); hA, hB = bp.halves(idx, seed=5)
        for nm, v in ((f"halfA_{k}", hA), (f"halfB_{k}", hB), (f"class_{k}", (hA + hB) / 2)):
            with mrcfile.new(O + nm + ".mrc", overwrite=True) as m: m.set_data(v.astype(np.float32)); m.voxel_size = bp.apix
    say("class maps written")
json.dump(dict(K=K, band=[LO, HI], null=NULL, start_sizes=np.bincount(lab0, minlength=K).tolist(), iters=hist, done=True), open(O + "summary.json", "w"), indent=1)
open(O + "done.json", "w").write("{}")
