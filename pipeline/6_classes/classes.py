"""Classes in the multi-coordinate space of the search.  MP_NC_CL (300) k-means cells in the weighted coordinates S * wt are back-projected
as half maps; the cells are grouped into K = 1..10 classes by n-weighted k-means on the CELL MAPS (k-means++, 20 restarts).  Per K:
  within  reproducible within-class variance  sum_k sum_{c in k} n_c <A_c - Abar_k, B_c - Bbar_k>  (/ total)  -> elbow
  cv      half-map cross-validation (classes fitted on A maps scored on B maps and vice versa)
  min_offblend  smallest reproducible own component <own_A, own_B> over the classes (/ the K=2 class difference)  -> off-blend halving rule
Usage: MP_OUT MP_ENV64 MP_STACK MP_POSES MP_CTF [MP_AP64 MP_NC_CL MP_KEEP MP_LP MP_GRID] classes.py
-> MP_OUT/mc/: MP_model_cells.npz (cell, A, B), MP_model_labels.npz (cell, K1..K10: class of each cell), MP.json"""
import json, os, sys
from pathlib import Path
import numpy as np
from sklearn.cluster import MiniBatchKMeans
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
from fastbp import FastBP
from maps import cropper, group_halves

E = os.environ; MO = E["MP_OUT"] + "/"; O = MO + "mc/"; os.makedirs(O, exist_ok=True); KM = 10; NC = int(E.get("MP_NC_CL", "300")); cf = O + "MP_model_cells.npz"
d = MO + "iter2/"; S = np.load(d + ("S_final.npy" if os.path.exists(d + "S_final.npy") else "S_current.npy")).astype(np.float64)
wt = np.load(d + "wt_current.npy") if os.path.exists(d + "wt_current.npy") else np.ones(1); S, wt = S[:, :len(wt)], wt[: S.shape[1]]
KEEP = int(E.get("MP_KEEP", "0"))                     # keep only the first KEEP coordinates (main + seed-reproducible rounds)
if KEEP: S, wt = S[:, :KEEP], wt[:KEEP]
if os.path.exists(cf):
    z = np.load(cf); cell, A, B = z["cell"], z["A"].astype(np.float64), z["B"].astype(np.float64)
else:
    bp = FastBP(E["MP_STACK"], E["MP_POSES"], E["MP_CTF"]); cell = MiniBatchKMeans(NC, random_state=0, n_init=3, batch_size=4096).fit_predict(S * wt)
    A, B = group_halves(bp, cell, NC, cropper(np.load(E["MP_ENV64"]), bp.apix * 2)); np.savez(cf, cell=cell, A=A.astype(np.float32), B=B.astype(np.float32))
n = np.bincount(cell, minlength=NC).astype(float); g = ((A + B) / 2).mean(); A /= g; B /= g; X = (A + B) / 2
cen = lambda M, m: (M[m] * n[m, None]).sum(0) / n[m].sum()


def kmaps(X, K, restarts=20):
    """n-weighted k-means of the cell maps (k-means++ seeding); best of `restarts` by the weighted within-class sum of squares"""
    best = None; W = n / n.sum()
    for r in range(restarts):
        rs = np.random.default_rng(r); c = [rs.choice(len(X), p=W)]
        for _ in range(1, K):
            dd = np.min([((X - X[i]) ** 2).sum(1) for i in c], 0) * W; c.append(rs.choice(len(X), p=dd / dd.sum()))
        C = X[c]
        for _ in range(100):
            lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1); C2 = np.array([cen(X, lab == k) if (lab == k).any() else C[k] for k in range(K)])
            if np.allclose(C2, C): break
            C = C2
        J = float((n * ((X - C[lab]) ** 2).sum(1)).sum())
        if best is None or J < best[0]: best = (J, lab.copy())
    return best[1]


def offspan(hA, hB, others):
    """reproducible part of a class map outside the affine span of the other classes"""
    Om = np.array(others); base = Om[0]; Bm = (Om[1:] - base).T
    r = lambda h: (h - base) - (Bm @ np.linalg.lstsq(Bm, h - base, rcond=None)[0] if Bm.shape[1] else 0)
    return float(r(hA) @ r(hB))


allm = np.ones(NC, bool); tot = float((n[:, None] * ((A - cen(A, allm)) * (B - cen(B, allm)))).sum()); within, cv, minoff, labs = [], [], {}, {}
for K in range(1, KM + 1):
    lab = kmaps(X, K) if K > 1 else np.zeros(NC, int); labs[K] = lab
    within.append(sum(float((n[lab == k, None] * (A[lab == k] - cen(A, lab == k)) * (B[lab == k] - cen(B, lab == k))).sum()) for k in range(K)))
    lA, lB = (kmaps(A, K, 8), kmaps(B, K, 8)) if K > 1 else (lab, lab)
    cv.append((sum(float((n[lA == k, None] * (B[lA == k] - cen(A, lA == k)) ** 2).sum()) for k in range(K)) + sum(float((n[lB == k, None] * (A[lB == k] - cen(B, lB == k)) ** 2).sum()) for k in range(K))) / 2)
    if K >= 3:
        full = [cen(X, lab == k) for k in range(K)]
        minoff[K] = min(offspan(cen(A, lab == k), cen(B, lab == k), [full[i] for i in range(K) if i != k]) for k in range(K))
w = np.array(within); xn = np.arange(KM) / (KM - 1); yn = (w - w[-1]) / (w[0] - w[-1]); elbow = int(np.argmax(1 - xn - yn)) + 1
un = max(((cen(A, labs[2] == 0) - cen(A, labs[2] == 1)) * (cen(B, labs[2] == 0) - cen(B, labs[2] == 1))).sum(), 1e-12); k_off = 2
if minoff.get(3, 0) / un > 0:
    k_off = 3
    while k_off < KM and minoff[k_off + 1] >= 0.5 * minoff[k_off]: k_off += 1
res = dict(n_coords=int(S.shape[1]), weights=[float(v) for v in wt], within=[round(v / tot, 4) for v in within], cv=[round(v, 1) for v in cv], cv_min=int(np.argmin(cv)) + 1,
           min_offblend={k: round(v / un, 4) for k, v in minoff.items()}, elbow=elbow, offblend_halving=k_off, sizes={K: [int(n[labs[K] == k].sum()) for k in range(K)] for K in range(2, 7)})
np.savez(O + "MP_model_labels.npz", cell=cell, **{f"K{K}": labs[K] for K in labs}); json.dump(dict(model=res), open(O + "MP.json", "w"), indent=1)
print("MC", json.dumps({k: v for k, v in res.items() if k != "sizes"}), flush=True)
