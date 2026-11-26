"""(11-25) Envelope test 2.  As env_test.py (heterorefinement classes reconstructed with consensus poses, class - consensus in two half sets, E = sum(dA*dB)), but:
half maps saved (od/envtest/<DS>_halves.npz) so later checks need no reconstruction; envelopes compared: 6MSM (current), 5UAK, union 6MSM+5UAK, consensus;
signal OUTSIDE the union envelope split along the membrane axis (fitted 6MSM, TMD -> NBD) into extracellular / membrane belt / cytoplasmic.  The belt is the slab
where the consensus envelope extends beyond the union envelope (slices with > 50 % of the peak excess), i.e. the micelle; all label-free (consensus + PDB only)."""
import json, os, sys, numpy as np, mrcfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../lib" if "pipeline" in __file__ else "../lib"))
from fastbp import FastBP, lowpass
B = os.environ.get("RES", "/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn") + "/model_pipeline/"; O = B + "od/envtest/"; E = B + "env/"
rows = {l.split()[0]: l.split() for l in open(B + "datasets.tsv") if l.strip() and not l.startswith("#")}; res = {}
for ds, K in (("J2708", 4), ("J399", 9)):
    sj = json.load(open(B + f"od/{ds}/hetero_K{K}/set.json")); lab = np.load(B + f"od/{ds}/hetero_K{K}/labels.npy"); names = sj["names"]; hf = O + f"{ds}_halves.npz"
    if os.path.exists(hf): z = np.load(hf); A, Bm, n = z["A"], z["B"], z["n"]
    else:
        bp = FastBP(*rows[ds][1:4]); ap = bp.apix * 2
        def crop(v):
            f = np.fft.fftshift(np.fft.fftn(v))[32:96, 32:96, 32:96]; return lowpass(np.real(np.fft.ifftn(np.fft.ifftshift(f))).astype(np.float32) / 8, ap, 10.0)
        A, Bm, n = [], [], []
        for k in range(K):
            idx = np.flatnonzero(lab == k); a, b = bp.halves(idx, seed=0); A.append(crop(a)); Bm.append(crop(b)); n.append(len(idx)); print(ds, names[k], len(idx), flush=True)
        A, Bm, n = np.array(A), np.array(Bm), np.array(n, float); np.savez(hf, A=A, B=Bm, n=n, names=names)
    w = n / n.sum(); cA = np.tensordot(w, A, 1); cB = np.tensordot(w, Bm, 1)
    g = np.indices((64, 64, 64)); sph = ((g - 32) ** 2).sum(0) < (0.45 * 64) ** 2
    envs = {k: np.load(E + f"{ds}/env64_{v}.npy").astype(bool) for k, v in (("6msm", "pdb"), ("5uak", "5uak"), ("union", "union"))}; envs["consensus"] = np.load(E + f"{ds}/env64.npy").astype(bool)
    ax = json.load(open(E + f"{ds}/axis.json")); a = np.array(ax["axis"]); c = np.array(ax["tmd_centre"])
    t = (g[2] - c[0]) * a[0] + (g[1] - c[1]) * a[1] + (g[0] - c[2]) * a[2]            # voxel (z,y,x) arrays; axis in (x,y,z)
    ex = envs["consensus"] & ~envs["union"]; tb = np.round(t).astype(int); prof = {int(s): int((ex & (tb == s)).sum()) for s in range(int(tb[sph].min()), int(tb[sph].max()) + 1)}
    pk = max(prof, key=prof.get); lo = hi = pk
    while prof.get(lo - 1, 0) > 0.5 * prof[pk]: lo -= 1
    while prof.get(hi + 1, 0) > 0.5 * prof[pk]: hi += 1
    reg = {"extracellular": t < lo - 0.5, "belt": (t >= lo - 0.5) & (t <= hi + 0.5), "cytoplasmic": t > hi + 0.5}
    out_u = sph & ~envs["union"]; res[ds] = dict(belt_slab=[lo, hi], profile=prof, classes={})
    print(f"{ds}: membrane axis t (voxels, + = toward NBDs); consensus-minus-union excess per slice:", " ".join(f"{s}:{v}" for s, v in prof.items() if v), "| belt slab", lo, hi, flush=True)
    for k in range(K):
        dA, dB = A[k] - cA, Bm[k] - cB; p = dA * dB; tot = float(p[sph].sum()); r = dict(n=int(n[k]))
        for en, M in envs.items(): r["out_" + en] = round(float(p[sph & ~M].sum()) / tot, 3)
        ou = float(p[out_u].sum()); r["outside_union_split"] = {rn: round(float(p[out_u & R].sum()) / ou, 3) if ou > 0 else None for rn, R in reg.items()}
        r["rep_outside_union"] = {rn: round(float(np.corrcoef(dA[out_u & R], dB[out_u & R])[0, 1]), 3) for rn, R in reg.items() if (out_u & R).sum() > 50}
        res[ds]["classes"][names[k]] = r
    json.dump(res, open(O + "envtest2.json", "w"), indent=1)
for ds, d in res.items():
    print(f"\n{ds}: share of class signal outside each envelope | of the signal outside the union: extracellular / belt / cytoplasmic (half-set rep)")
    for cn, r in d["classes"].items():
        s = r["outside_union_split"]; rp = r["rep_outside_union"]
        print(f"  {cn} n={r['n']:6d} 6msm {r['out_6msm']:.2f} 5uak {r['out_5uak']:.2f} union {r['out_union']:.2f} consensus {r['out_consensus']:.2f} | "
              + " / ".join(f"{s[x]:.2f} ({rp.get(x, float('nan')):.2f})" for x in ("extracellular", "belt", "cytoplasmic")))
