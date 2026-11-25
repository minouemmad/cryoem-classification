"""First pass: sharpen the main coordinate with the residual coefficients.
PC1 (volume-PC scores) -> 24 equal-count bins (half maps) -> end maps e_lo / e_hi = mean of the 2 end bins of each side ->
MP_NC_SH (200) k-means cells in standardised (PC1, z_1..z_8) back-projected -> t_c = projection of each cell map on the e_lo -> e_hi line ->
weighted least squares t ~ 1 + PC1 + z fitted on half A (scored on half B) and vice versa -> s = mean of the two fits for every particle.
Env: MP_OUT MP_SCORES MP_Z MP_ENV64 MP_STACK MP_POSES MP_CTF [MP_LP MP_GRID]  -> MP_OUT/e2e_s.npy, e2e_sharpen.json, unmix_{pc1,s}_eq*.npz"""
import json, os, sys
from pathlib import Path
import numpy as np
from sklearn.cluster import MiniBatchKMeans
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
from fastbp import FastBP
from maps import cropper, group_halves

E = os.environ; O = E["MP_OUT"] + "/"
S = np.load(E["MP_SCORES"])[:, 0].astype(np.float64); N = len(S); Z = np.load(E["MP_Z"]).astype(np.float64); assert len(Z) == N, (len(Z), N)
bp = FastBP(E["MP_STACK"], E["MP_POSES"], E["MP_CTF"]); crop = cropper(np.load(E["MP_ENV64"]), bp.apix * 2)


def bins(coord, B, name):
    """B equal-count bins along coord -> half maps (saved for inspection)"""
    order = np.argsort(coord, kind="stable"); lab = np.empty(N, int); rk = np.argsort(order) / N
    for i in range(B): lab[order[int(i * N / B): int((i + 1) * N / B)]] = i
    HA, HB = group_halves(bp, lab, B, crop)
    np.savez(O + name, A=HA.astype(np.float32), B=HB.astype(np.float32), n=np.bincount(lab, minlength=B), rank_mid=np.array([rk[lab == b].mean() for b in range(B)]))
    return HA, HB


HA, HB = bins(S, 24, "unmix_pc1_eq24.npz"); X = (HA + HB) / 2; e_lo = X[:2].mean(0); u = X[-2:].mean(0) - e_lo; un = u @ u
NC = int(E.get("MP_NC_SH", "200")); F = np.column_stack([S, Z]); F = (F - F.mean(0)) / F.std(0)
c = MiniBatchKMeans(NC, random_state=0, n_init=3, batch_size=4096).fit_predict(F); n = np.bincount(c, minlength=NC)
CA, CB = group_halves(bp, c, NC, crop); tA, tB = (CA - e_lo) @ u / un, (CB - e_lo) @ u / un
D1 = np.column_stack([np.ones(NC), np.array([F[c == k].mean(0) for k in range(NC)])]); w = np.sqrt(n)
fit = lambda cols, y: np.linalg.lstsq(D1[:, cols] * w[:, None], y * w, rcond=None)[0]
r2 = lambda cols, b, y: float(1 - (n * (y - D1[:, cols] @ b) ** 2).sum() / (n * (y - np.average(y, weights=n)) ** 2).sum())
allc, pc = list(range(D1.shape[1])), [0, 1]; bA, bB = fit(allc, tA), fit(allc, tB)
out = dict(r2_pc1_only=[r2(pc, fit(pc, tA), tB), r2(pc, fit(pc, tB), tA)], r2_pc1_plus_residual=[r2(allc, bA, tB), r2(allc, bB, tA)],
           beta_halfA=[round(float(x), 4) for x in bA], beta_halfB=[round(float(x), 4) for x in bB])
s = np.column_stack([np.ones(N), F]) @ ((bA + bB) / 2); np.save(O + "e2e_s.npy", s.astype(np.float32)); out["corr_s_pc1"] = float(np.corrcoef(s, S)[0, 1])
json.dump(out, open(O + "e2e_sharpen.json", "w"), indent=1); print("SH", json.dumps(out), flush=True)
bins(s, 24, "unmix_s_eq24.npz"); bins(s, 48, "unmix_s_eq48.npz")
