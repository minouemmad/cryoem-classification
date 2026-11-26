"""(11-25) Union protein envelope from fit_6msm.pdb + fit_5uak.pdb (consensus only, no heterorefinement info), same construction as mp_regions.py
(atoms -> 1.5 voxel Gaussian, > 10 % of max, dilated 2 voxels, flipped back to the pipeline frame when the best hand was 1).  Also the membrane axis
(TMD -> NBD centroid of fitted 6MSM) for the micelle-belt split.  Writes env64_union.npy, env64_5uak.npy, env64_6msmT.npy, axis.json in env/<DS>/."""
import os
import json, re, sys, numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation
E = os.environ.get("RES", "/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn") + "/model_pipeline/env/"
for ds in sys.argv[1:]:
    d = E + ds + "/"; A = float(open(d + "apix.txt").read()); N = 64; hand = int(re.search(r"hand (\d)", open(d + "regions.log").read()).group(1))
    m = np.load(d + "consensus64.npy"); m = m[::-1] if hand else m
    def atoms(f): return np.array([(int(l[22:26]) if l[21] == "A" else -1, float(l[30:38]), float(l[38:46]), float(l[46:54])) for l in open(f) if l.startswith("ATOM")])   # non-A chains (R-domain fragments): density only
    def dens(xyz):
        g = np.zeros((N, N, N), np.float32); i = np.rint(xyz / A).astype(int); i = i[((i >= 0) & (i < N)).all(1)]; np.add.at(g, (i[:, 2], i[:, 1], i[:, 0]), 1.0); return gaussian_filter(g, 1.5)
    env, info = {}, {}
    for nm in ("6msm", "5uak"):
        r = atoms(d + f"fit_{nm}.pdb"); g = dens(r[:, 1:]); p = g > g.max() * 0.10; mk = g > g.max() * 0.05
        tmd = (r[:, 0] >= 1) & (r[:, 0] <= 380) | ((r[:, 0] >= 840) & (r[:, 0] <= 1172)); nbd = ((r[:, 0] >= 381) & (r[:, 0] <= 645)) | (r[:, 0] >= 1173)
        info[nm] = dict(corr=round(float(np.corrcoef(g[mk], m[mk])[0, 1]), 3), nbd_sep=round(float(np.linalg.norm(r[(r[:, 0] >= 381) & (r[:, 0] <= 645), 1:].mean(0) - r[r[:, 0] >= 1173, 1:].mean(0))), 1))
        if nm == "6msm": c0 = r[tmd, 1:].mean(0) / A; c1 = r[nbd, 1:].mean(0) / A
        e = binary_dilation(p, iterations=2); env[nm] = e[::-1].copy() if hand else e
    ax = (c1 - c0) / np.linalg.norm(c1 - c0); ctr = c0
    if hand: ax = ax * np.array([1, 1, -1]); ctr = ctr.copy(); ctr[2] = N - 1 - ctr[2]   # (x, y, z) voxel coords; z index flipped
    u = env["6msm"] | env["5uak"]; np.save(d + "env64_union.npy", u); np.save(d + "env64_5uak.npy", env["5uak"]); np.save(d + "env64_6msmT.npy", env["6msm"])
    old = np.load(d + "env64_pdb.npy"); c = np.load(d + "env64.npy")
    json.dump(dict(axis=ax.tolist(), tmd_centre=ctr.tolist()), open(d + "axis.json", "w"))
    print("UNION", ds, "hand", hand, info, "| voxels old6msm", int(old.sum()), "6msmT", int(env["6msm"].sum()), "5uak", int(env["5uak"].sum()), "union", int(u.sum()),
          "| old∩6msmT", int((old & env["6msm"]).sum()), "| consensus env", int(c.sum()), "outside union", int((c & ~u).sum()), flush=True)
