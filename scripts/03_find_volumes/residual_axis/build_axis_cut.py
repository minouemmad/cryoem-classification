"""Axis-cut ladder + RANDOM-DIRECTION CONTROL. The 4 residual coefficients in the middle window (ranks 0.30-0.70) form an almost round cloud (covariance eigenvalues 1.15 / 0.98 / 0.95 / 0.92), and the ma"""
import json
from pathlib import Path
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as L
A = Path("/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn/J4001_gP25_WT/candidate_groups_epoch50/axis_survey")
S = np.load(A / "pc_scan_J4001_rae_nuposes.scores.npy"); Z = np.load(A / "ppca_real8_z.npy"); N = len(S); rank = np.argsort(np.argsort(S[:, 0])) / N
mid = np.where((rank >= 0.30) & (rank < 0.70))[0]; Zs = (Z[mid, :4] - Z[mid, :4].mean(0)) / Z[mid, :4].std(0)
g0 = np.load(A / "set_win_resid/groups/leaves_K2_g0.npy"); y = np.isin(mid, g0).astype(int); w = L().fit(Zs, y).coef_[0]; w /= np.linalg.norm(w)
if (Zs @ w)[y == 1].mean() > (Zs @ w)[y == 0].mean(): w = -w                                   # low u = manual P7-like half
rng = np.random.default_rng(5); G = []; I = []
u = Zs @ w
for side in ("low", "high"):
    for q in (0.3, 0.4, 0.5, 0.6, 0.7):
        t = np.quantile(u, q); g = mid[u < t] if side == "low" else mid[u >= t]; G.append(g); I.append(dict(axis="gmm", side=side, q=q))
for r in range(3):
    v = rng.normal(size=4); v /= np.linalg.norm(v); ur = Zs @ v; t = np.median(ur)
    for side in ("low", "high"): G.append(mid[ur < t] if side == "low" else mid[ur >= t]); I.append(dict(axis=f"random{r + 1}", side=side, q=0.5, direction=np.round(v, 3).tolist()))
d = A / "set_axiscut" / "groups"; d.mkdir(parents=True, exist_ok=True)
for j, (g, i) in enumerate(zip(G, I)):
    np.save(d / f"leaves_K16_g{j}.npy", np.sort(g)); print(j, i["axis"], i["side"], i["q"], len(g), "share in manual P7-like half %.2f" % np.isin(g, g0).mean())
json.dump(dict(kind="axis cut ladder + random-direction control", w=w.tolist(), windows=[dict(j=j, K=16, n=int(len(g)), **i) for j, (g, i) in enumerate(zip(G, I))]), open(A / "set_axiscut" / "windows.json", "w"), indent=1)
