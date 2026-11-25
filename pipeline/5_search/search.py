"""Fully conditioned iterative search for further variation, starting from the sharpened coordinate s.  Each round r:
 1. 64 equal-count cells conditioned on ALL current coordinates S (recursive bisection along the leading direction of S * wt);
    the residual model (4_residual/residual_ppca.py) is refitted on images minus their cell map -> coefficients z (conditional on S);
 2. MP_NC_IT (400) k-means cells in (S * wt, 0.5 z) are back-projected as half maps A, B (envelope, MP_LP A);
 3. what S already predicts is removed: each cell's maps and mean z minus the n-weighted mean of its 8 nearest cells in S;
    the direction w in z with the largest reproducible map effect solves  (C_A C_B^T) w = lambda G w  (C_X = sum_c n_c f_c R_X,c ; G = sum_c n_c f_c f_c^T);
    null: z permuted between cells of similar S (200 permutations); p = P(lambda_null >= lambda);
 4. the map effect of w is split into the part along the tangents of S (it sharpens them: S += x beta) and the rest; if the rest
    is also outside its null (p_off <= 0.05) x = w.z becomes a NEW coordinate with weight sqrt(lambda / T_main).  No orthogonality imposed.
 Stop when p > 0.05 or after MAXR rounds.  Seed-reproducibility of each round's mode map decides how many rounds to keep (6_classes/repro.py).
Usage: MP_OUT=<run dir> MP_ENV64 MP_STACK MP_POSES MP_CTF [MP_K MP_HI MP_LO MP_NC_IT MP_LP MP_GRID]  search.py [MAXR=10]
-> MP_OUT/iter2/: S_final.npy, wt_current.npy, iter.json, mode_r*.npy (map effect of each round), x_r*.npy"""
import json, os, re, subprocess, sys
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import MiniBatchKMeans
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
from fastbp import FastBP
from maps import cropper, group_halves, up2

E = os.environ; O = Path(E["MP_OUT"]) / "iter2"; O.mkdir(parents=True, exist_ok=True); MAXR = int(sys.argv[1]) if len(sys.argv) > 1 else 10
PPCA = str(Path(__file__).resolve().parents[1] / "4_residual/residual_ppca.py"); rng = np.random.default_rng(0)
env64 = np.load(E["MP_ENV64"]); np.save(O / "env.npy", up2(env64))
std = lambda a: (a - a.mean(0)) / a.std(0)
S = std(np.load(E["MP_OUT"] + "/e2e_s.npy").astype(np.float64)[:, None]); N = len(S)
bp = FastBP(E["MP_STACK"], E["MP_POSES"], E["MP_CTF"]); crop = cropper(env64, bp.apix * 2)


def cond_cells(S, wt, depth=6):
    """2^depth equal-count cells by recursive bisection along the leading principal direction of the weighted coordinates"""
    X = S * wt; cells = [np.arange(len(S))]
    for _ in range(depth):
        new = []
        for i in cells:
            Y = X[i] - X[i].mean(0); v = np.linalg.svd(Y[:: max(1, len(i) // 20000)], full_matrices=False)[2][0] if X.shape[1] > 1 else np.ones(1)
            o = np.argsort(Y @ v, kind="stable"); h = len(i) // 2; new += [i[o[:h]], i[o[h:]]]
        cells = new
    c = np.empty(len(S), int)
    for k, i in enumerate(cells): c[i] = k
    return c


wt = np.ones(1); Tmain = None; log = []
for rnd in range(1, MAXR + 1):
    # 1. residual model conditioned on the current coordinates (cells passed as a rank coordinate: column 0 sorted by cell)
    cc = cond_cells(S, wt); order = np.lexsort((rng.random(N), cc)); sc = np.empty(N); sc[order] = np.arange(N); np.save(O / f"cond_r{rnd}.npy", sc[:, None]); T = f"r{rnd}"
    e = dict(E, MCR_STACK=E["MP_STACK"], MCR_POSES=E["MP_POSES"], MCR_CTF=E["MP_CTF"], MCR_SCORES=str(O / f"cond_r{rnd}.npy"), MCR_NBM="64", OMP_NUM_THREADS="8")
    r = subprocess.run([sys.executable, PPCA, str(O), T, E.get("MP_K", "8"), "30", E.get("MP_HI", "7.0"), E.get("MP_LO", "50.0")], env=e, capture_output=True, text=True)
    open(O / f"ppca_{T}.stdout", "w").write(r.stdout + r.stderr)
    cosl = re.findall(r"fold-vs-fold subspace cosines \(principal angles\): (.*)", r.stdout); cos = [float(x) for x in cosl[-1].split()] if cosl else None
    z = std(np.load(O / f"ppca_{T}_z.npy").astype(np.float64))
    # 2. cells in (S, z) -> half maps
    NC = int(E.get("MP_NC_IT", "400")); c = MiniBatchKMeans(NC, random_state=rnd, n_init=3, batch_size=4096).fit_predict(np.column_stack([S * wt, 0.5 * z]))
    n = np.bincount(c, minlength=NC).astype(float); A, B = group_halves(bp, c, NC, crop); g = ((A + B) / 2).mean(); A /= g; B /= g
    Sc = np.array([S[c == k].mean(0) for k in range(NC)]); zc = np.array([z[c == k].mean(0) for k in range(NC)])
    # 3. remove what S predicts (8 nearest cells in S), then the most reproducible direction in z
    d = (((Sc[:, None] - Sc[None]) * wt) ** 2).sum(-1); np.fill_diagonal(d, np.inf); nn = np.argsort(d, 1)[:, :8]; Wn = np.zeros((NC, NC))
    for i in range(NC): Wn[i, nn[i]] = n[nn[i]] / n[nn[i]].sum()
    RA, RB, fz = A - Wn @ A, B - Wn @ B, zc - Wn @ zc; Ttot = float((n * (RA * RB).sum(1)).sum())

    def solve(f):
        CA, CB = (f * n[:, None]).T @ RA, (f * n[:, None]).T @ RB; M = CA @ CB.T; G = (f * n[:, None]).T @ f; lam, V = eigh((M + M.T) / 2, G)
        return lam[-1], V[:, -1], CA, CB, G
    if Tmain is None: Tmain = float((n * ((A - (n @ A) / n.sum()) * (B - (n @ B) / n.sum())).sum(1)).sum())      # reproducible between-cell variance (first round = main axis)
    lam, w, CA, CB, G = solve(fz); blk = np.argsort(Sc[:, 0]).reshape(-1, 10)

    def perm():
        p = np.arange(NC)
        for b_ in blk: p[b_] = rng.permutation(b_)
        return fz[p]
    # 4. split the map effect into tangent (sharpen S) and off-tangent (new coordinate) parts
    D1 = np.column_stack([np.ones(NC), Sc]) * np.sqrt(n)[:, None]; tang = np.linalg.lstsq(D1, (A + B) / 2 * np.sqrt(n)[:, None], rcond=None)[0][1:]

    def split(w_, CA_, CB_, G_):
        q = float(w_ @ G_ @ w_); vA, vB = CA_.T @ w_ / q, CB_.T @ w_ / q; beta = np.linalg.lstsq(tang.T, (vA + vB) / 2, rcond=None)[0]
        P = lambda v: tang.T @ np.linalg.lstsq(tang.T, v, rcond=None)[0]
        return float(vA @ vB), float((vA - P(vA)) @ (vB - P(vB))) * q, beta, (vA + vB) / 2
    tot, off, beta, mode = split(w, CA, CB, G); nl, no = [], []
    for _ in range(200):
        f = perm(); l_, w_, CA_, CB_, G_ = solve(f); nl.append(l_); no.append(split(w_, CA_, CB_, G_)[1])
    p = float((np.array(nl) >= lam).mean()); p_off = float((np.array(no) >= off).mean()); xr = z @ w; xr = xr - xr.mean(); x = xr / xr.std()
    rec = dict(round=rnd, n_coords=S.shape[1], pct_of_main=float(100 * lam / Tmain), weights=[float(v) for v in wt], fold_cosines=cos, total_conditional_reproducible=Ttot,
               lam=float(lam), lam_share=float(lam / Ttot) if Ttot > 0 else None, null95=float(np.percentile(nl, 95)), p=p, off_tangent=off, off_null95=float(np.percentile(no, 95)),
               p_off=p_off, off_share_of_lam=float(off / lam) if lam > 0 else None, beta=[float(b) for b in beta], corr_with_coords=[float(np.corrcoef(x, S[:, j])[0, 1]) for j in range(S.shape[1])])
    np.save(O / f"mode_r{rnd}.npy", mode.astype(np.float32)); np.save(O / f"x_r{rnd}.npy", x.astype(np.float32))
    if p > 0.05:
        rec["action"] = "stop: no reproducible conditional variation"; log.append(rec); print("IT", json.dumps(rec), flush=True); break
    S = S + np.outer(xr, beta); act = ["sharpened existing coordinates"]
    if p_off <= 0.05: S = np.column_stack([S, x]); wt = np.append(wt, np.sqrt(max(lam, 0) / Tmain)); act.append("added a new coordinate")
    S = std(S); np.save(O / "S_current.npy", S.astype(np.float32)); np.save(O / "wt_current.npy", wt)
    rec["action"] = "; ".join(act); log.append(rec); json.dump(log, open(O / "iter_partial.json", "w"), indent=1); print("IT", json.dumps(rec), flush=True)
np.save(O / "S_final.npy", S.astype(np.float32)); np.save(O / "wt_current.npy", wt); json.dump(log, open(O / "iter.json", "w"), indent=1); print("IT done", S.shape, flush=True)
