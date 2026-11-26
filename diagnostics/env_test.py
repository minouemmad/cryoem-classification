"""(11-25) Envelope test for the wide-V heterorefinement classes (J2708 C09, J399 C11): reconstruct every heterorefinement class from its own particles with the CONSENSUS
poses (what the pipeline uses), full 64^3 box, 10 A.  Per class: difference from the consensus (particle-weighted mean of the classes) in two independent half sets;
signal energy E = sum(dA*dB) (noise-unbiased) inside vs outside the PDB envelope (pipeline rows J2708 / J399) and the consensus envelope (rows J2708c / J399c),
within a sphere of radius 0.45 box; half-set reproducibility corr(dA, dB) inside / outside.  Writes od/envtest/<DS>_<class>_{map,diff}.mrc + envtest.json."""
import json, os, sys, numpy as np, mrcfile
from scipy.ndimage import binary_dilation
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../lib" if "pipeline" in __file__ else "../lib"))
from fastbp import FastBP, lowpass
B = os.environ.get("RES", "/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn") + "/model_pipeline/"; O = B + "od/envtest/"; os.makedirs(O, exist_ok=True)
rows = {l.split()[0]: l.split() for l in open(B + "datasets.tsv") if l.strip() and not l.startswith("#")}
res = {}
for ds, K in (("J2708", 4), ("J399", 9)):
    sj = json.load(open(B + f"od/{ds}/hetero_K{K}/set.json")); lab = np.load(B + f"od/{ds}/hetero_K{K}/labels.npy"); names = sj["names"]
    _, stk, pos, ctf, env64 = rows[ds][:5]; envc = rows[ds + "c"][4]
    bp = FastBP(stk, pos, ctf); ap = bp.apix * 2
    def crop(v):
        f = np.fft.fftshift(np.fft.fftn(v)); f = f[32:96, 32:96, 32:96]; return lowpass(np.real(np.fft.ifftn(np.fft.ifftshift(f))).astype(np.float32) / 8, ap, 10.0)
    A, Bm, n = [], [], []
    for k in range(K):
        idx = np.flatnonzero(lab == k); a, b = bp.halves(idx, seed=0); A.append(crop(a)); Bm.append(crop(b)); n.append(len(idx)); print(ds, names[k], len(idx), flush=True)
    A, Bm, n = np.array(A), np.array(Bm), np.array(n, float); w = n / n.sum(); cA = np.tensordot(w, A, 1); cB = np.tensordot(w, Bm, 1)
    g = np.indices((64, 64, 64)) - 32; sph = (g ** 2).sum(0) < (0.45 * 64) ** 2
    envs = {"pdb": np.load(env64).astype(bool), "consensus": np.load(envc).astype(bool)}; envs["pdb+8A"] = binary_dilation(envs["pdb"], iterations=2)
    res[ds] = {}
    for k in range(K):
        dA, dB = A[k] - cA, Bm[k] - cB; r = dict(n=int(n[k]))
        for en, E in envs.items():
            ins, out = sph & E, sph & ~E; e_in, e_out = float((dA * dB)[ins].sum()), float((dA * dB)[out].sum())
            r[en] = dict(frac_outside=round(e_out / (e_in + e_out), 3) if e_in + e_out > 0 else None,
                         rep_in=round(float(np.corrcoef(dA[ins], dB[ins])[0, 1]), 3), rep_out=round(float(np.corrcoef(dA[out], dB[out])[0, 1]), 3))
        res[ds][names[k]] = r
        for nm, v in (("map", (A[k] + Bm[k]) / 2), ("diff", (dA + dB) / 2)):
            with mrcfile.new(O + f"{ds}_{names[k]}_{nm}.mrc", overwrite=True) as m: m.set_data(v.astype(np.float32)); m.voxel_size = ap
    for en, E in envs.items():
        with mrcfile.new(O + f"{ds}_env_{en}.mrc", overwrite=True) as m: m.set_data(E.astype(np.float32)); m.voxel_size = ap
    json.dump(res, open(O + "envtest.json", "w"), indent=1)
for ds, d in res.items():
    print(f"\n{ds}: share of class-specific signal OUTSIDE the envelope (half-set reproducibility inside / outside)")
    for c, r in d.items():
        print(f"  {c} n={r['n']:6d} " + "  ".join(f"{en}: {r[en]['frac_outside']:.2f} ({r[en]['rep_in']:.2f}/{r[en]['rep_out']:.2f})" for en in ("pdb", "pdb+8A", "consensus")))
