"""Residual-heterogeneity model, label-free: a K-component linear subspace of what PC1 does not explain, fitted to the IMAGES (3DVA-style probabilistic PCA by EM; fixed NU poses, CTF, per-image scale, pe
Usage: residual_fit.py OUT TAG [STACK=real|sim] [K=8] [ITERS=30] [HI=7.0] [LO=50.0]     |    residual_fit.py selftest"""
import os, sys, time
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as Fnn
from cryodrgn import fft


def projK(U, rot, coords, D):
    """central slices of K Hartley volumes U (K,D,D,D) at rotations rot (B,3,3) -> (B,K,M)"""
    d2 = D // 2
    g = (((coords @ rot).clamp(-d2, d2) + d2) / (D - 1) * 2 - 1)[None, None]
    return Fnn.grid_sample(U[None], g, mode="bilinear", align_corners=True)[0, :, 0].permute(1, 0, 2)


def scatter(acc, rot, vals, coords, D):
    """acc (D^3, C) += exact adjoint of the trilinear slicing: vals (B,M,C) placed at coords @ rot"""
    d2 = D // 2
    x = (coords @ rot).clamp(-d2, d2) + d2
    f = x.floor().clamp(max=D - 2); t = x - f; f = f.long(); v = vals.reshape(-1, vals.shape[-1])
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                w = (t[..., 0] if dx else 1 - t[..., 0]) * (t[..., 1] if dy else 1 - t[..., 1]) * (t[..., 2] if dz else 1 - t[..., 2])
                idx = ((f[..., 2] + dz) * D + f[..., 1] + dy) * D + f[..., 0] + dx
                acc.index_add_(0, idx.reshape(-1), v * w.reshape(-1, 1))


def realmask(U, msk, vox):
    """multiply each Hartley volume by the soft real-space mask; keep only the Fourier voxels vox (flat indices)"""
    K, D = U.shape[0], U.shape[1]; out = torch.zeros_like(U)
    for k in range(K):
        out[k, :-1, :-1, :-1] = fft.htn_center(fft.ihtn_center(U[k, :-1, :-1, :-1]) * msk)
    keep = torch.zeros(D ** 3, dtype=torch.bool, device=U.device); keep[vox] = True
    return out * keep.view(1, D, D, D)


def orthonorm(U):
    """U (K,D,D,D) plain-EM axes -> orthonormal Q ordered by model variance tau2 (eigen-decomposition of the K x K Gram matrix)"""
    K = U.shape[0]; F = U.reshape(K, -1).double(); ev, A = torch.linalg.eigh(F @ F.T); o = torch.argsort(ev, descending=True); ev = ev[o].clamp(min=1e-30); A = A[:, o]
    Q = ((A.T @ F) / ev.sqrt()[:, None]).float().reshape(U.shape)
    return Q, ev.float()


@torch.no_grad()
def estep(R, Cw, ROT, rows, Q, tau2, coords, D, bs=1000):
    """posterior mean zh (n,K) and covariance Sig (n,K,K) of the unit-prior coefficients; the coefficient along q_k is tau_k * zh_k"""
    K = Q.shape[0]; n = len(rows); zh = torch.empty((n, K), device=R.device); Sig = torch.empty((n, K, K), device=R.device)
    I = torch.eye(K, device=R.device); tau = tau2.sqrt()
    for s in range(0, n, bs):
        rr = rows[s:s + bs]; Wm = tau[None, :, None] * Cw[rr][:, None, :] * projK(Q, ROT[rr], coords, D)
        Sg = torch.linalg.inv(Wm @ Wm.transpose(1, 2) + I); Sig[s:s + bs] = Sg
        zh[s:s + bs] = (Sg @ (Wm * R[rr][:, None, :]).sum(-1)[..., None])[..., 0]
    return zh, Sig


@torch.no_grad()
def mstep(R, Cw, ROT, rows, zh, Sig, coords, D, vox, lam=0.1, bs=500):
    K = zh.shape[1]; iu = torch.triu_indices(K, K, device=R.device); acc = torch.zeros((D ** 3, K + iu.shape[1]), device=R.device)
    for s in range(0, len(rows), bs):
        rr = rows[s:s + bs]; c = Cw[rr]; z = zh[s:s + bs]; Ez = z[:, :, None] * z[:, None, :] + Sig[s:s + bs]
        scatter(acc, ROT[rr], torch.cat([(c * R[rr])[:, :, None] * z[:, None, :], (c * c)[:, :, None] * Ez[:, iu[0], iu[1]][:, None, :]], -1), coords, D)
    a = acc[vox]; del acc
    L = torch.zeros((len(vox), K, K), device=R.device); L[:, iu[0], iu[1]] = a[:, K:]; L[:, iu[1], iu[0]] = a[:, K:]
    reg = lam * L.diagonal(dim1=1, dim2=2).mean()
    u = torch.linalg.solve(L + reg * torch.eye(K, device=R.device), a[:, :K])
    U = torch.zeros((K, D ** 3), device=R.device); U[:, vox] = u.T
    return U.view(K, D, D, D)


@torch.no_grad()
def fit(R, Cw, ROT, rows, K, iters, coords, D, vox, msk, seed, say=print):
    g = torch.Generator(device="cpu").manual_seed(seed); n = len(rows)
    zh = torch.randn((n, K), generator=g).to(R.device); Sig = torch.zeros((n, K, K), device=R.device); Qp = None
    for it in range(iters):
        Q, tau2 = orthonorm(realmask(mstep(R, Cw, ROT, rows, zh, Sig, coords, D, vox), msk, vox))
        zh, Sig = estep(R, Cw, ROT, rows, Q, tau2, coords, D)
        shrink = 1 - Sig.diagonal(dim1=1, dim2=2).mean(0)          # mean posterior information per axis (0 = prior only)
        if Qp is not None:
            sv = torch.linalg.svdvals(Qp.reshape(K, -1).double() @ Q.reshape(K, -1).double().T)
            say(f"   iter {it + 1:2d}: subspace cosines vs previous iterate min {float(sv.min()):.4f} mean {float(sv.mean()):.4f} | tau2 "
                + " ".join(f"{x:.3g}" for x in tau2.tolist()) + " | information " + " ".join(f"{x:.3f}" for x in shrink.tolist()))
        Qp = Q
    return Q, tau2


def band_voxels(D, rmax, device):
    ax = torch.arange(D, device=device) - D // 2; zz, yy, xx = torch.meshgrid(ax, ax, ax, indexing="ij")
    return torch.nonzero(((xx ** 2 + yy ** 2 + zz ** 2).float().sqrt() <= rmax).reshape(-1))[:, 0]


def selftest():
    """planted 2-component subspace, weak per-image signal, CPU: the fit must recover the subspace and the out-of-sample coefficients"""
    from cryodrgn.lattice import Lattice
    torch.manual_seed(0); torch.set_num_threads(1); D = 33; d2 = D // 2; dev = torch.device("cpu")
    lat = Lattice(D, extent=d2, device=dev); m2 = lat.get_circular_mask(10); coords = lat.coords[m2]; M = coords.shape[0]
    ax = torch.arange(D - 1) - d2; zz, yy, xx = torch.meshgrid(ax, ax, ax, indexing="ij")
    blob = lambda c, s: torch.exp(-((xx - c[0]) ** 2 + (yy - c[1]) ** 2 + (zz - c[2]) ** 2) / (2 * s * s))
    msk = (blob((0, 0, 0), 7.0) > 0.3).float(); vox = band_voxels(D, 11.5, dev)
    tr = [blob((4, 0, 0), 2.0) - blob((-3, 2, 0), 2.0), blob((0, 4, 3), 2.5) - blob((0, -4, -2), 2.0)]
    Ut = torch.zeros((2, D, D, D))
    for k in range(2): Ut[k, :-1, :-1, :-1] = fft.htn_center(tr[k] * msk)
    Ut = realmask(Ut, msk, vox); Qt, _ = orthonorm(Ut)
    n = 12000; q = torch.randn(n, 4); q = q / q.norm(dim=1, keepdim=True); w, x, y, z = q.T
    ROT = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                       2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], 1).view(n, 3, 3)
    Cw = 0.5 + torch.rand(n, 1) * torch.ones(1, M); zt = torch.randn(n, 2) * torch.tensor([9.0, 6.0])
    sig = torch.cat([(Cw[s:s + 500] * (projK(Qt, ROT[s:s + 500], coords, D) * zt[s:s + 500, :, None]).sum(1)) for s in range(0, n, 500)])
    R = sig + torch.randn(n, M)
    print(f"selftest: {n} images, {M} pixels, per-image signal/noise power {float(sig.pow(2).mean()):.3f}")
    tr_rows, te_rows = torch.arange(0, n // 2), torch.arange(n // 2, n)
    Q, tau2 = fit(R, Cw, ROT, tr_rows, 3, 12, coords, D, vox, msk, 0)
    sv = torch.linalg.svdvals(Qt.reshape(2, -1).double() @ Q[:2].reshape(2, -1).double().T)
    zh, _ = estep(R, Cw, ROT, te_rows, Q, tau2, coords, D); zo = (zh * tau2.sqrt())[:, :2].double(); zt2 = zt[te_rows].double()
    cc = np.corrcoef(np.hstack([zo.numpy(), zt2.numpy()]).T)[:2, 2:]
    print("true-vs-fitted subspace cosines (first 2 axes):", sv.numpy().round(3), "| tau2", tau2.numpy().round(2), "(planted 81, 36, 0)")
    print("held-out coefficient correlation with the planted ones:\n", cc.round(3))
    assert sv.min() > 0.9 and np.abs(cc).max(1).min() > 0.5 and tau2[2] < 0.5 * tau2[1], "selftest FAILED"
    print("selftest OK")


if __name__ == "__main__":
    if sys.argv[1] == "selftest":
        selftest(); sys.exit()
    SC = os.environ.get("MCR_SC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib")); sys.path.insert(0, SC)      # MCR_SC / MCR_W: other machine (r-lnx209)
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
    from fastbp import FastBP, lowpass
    from scipy.ndimage import gaussian_filter
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.mixture import GaussianMixture
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    W = Path(os.environ.get("MCR_W", "/Dedicated/schnieders/maemmad/cryoem-classification/results_cryodrgn/J4001_gP25_WT")); A = Path(os.environ.get("MCR_A", str(W / "candidate_groups_epoch50/axis_survey")))
    O = Path(sys.argv[1]); TAG = sys.argv[2]; ar = sys.argv[3:] + [None] * 5
    STACK = ar[0] or "real"; K = int(ar[1] or 8); IT = int(ar[2] or 30); HI = float(ar[3] or 7.0); LO = float(ar[4] or 50.0); NULL = STACK == "sim"
    import os                                                    # (2026-11-20) optional switches, defaults = the original run
    FOLDSEED = int(os.environ.get("MCR_FOLDSEED", 0))            # random 2-fold split of the particles
    SCORES = os.environ.get("MCR_SCORES", "pc_scan_J4001_rae_nuposes.scores.npy")
    NOISENULL = os.environ.get("MCR_NOISENULL", "") == "1"
    NOEVAL = os.environ.get("MCR_NOEVAL", "") == "1"
    ENV = np.load(O / "env.npy"); nv = ENV.sum(); lab = None if NOEVAL else np.load(O / "j2923_labels.npy")
    if NOEVAL: O_REF = None
    ref = np.load(O / ("ref_j2923.npz" if not NOEVAL else "env.npy")) if not NOEVAL else None
    if NOEVAL: RA = RB = np.zeros((4, int(nv)))
    else: RA, RB = ref["A"].astype(np.float64), ref["B"].astype(np.float64)
    dist = lambda XA, XB, YA, YB: float(((XA - YA) * (XB - YB)).sum() / nv); d86 = dist(RA[2], RB[2], RA[0], RB[0])
    uu = (RA[0] + RB[0] - RA[2] - RB[2]) / 2; un = uu @ uu; t7 = ((RA[1] + RB[1] - RA[2] - RB[2]) / 2 @ uu) / un
    ownA = RA[1] - (RA[2] + t7 * (RA[0] - RA[2])); ownB = RB[1] - (RB[2] + t7 * (RB[0] - RB[2])); own2 = ownA @ ownB; own = (ownA + ownB) / 2
    alt = None if NOEVAL else (RA[3] + RB[3]) / 2 - (np.load(O / "all.npz")["A"][0] + np.load(O / "all.npz")["B"][0]) / 2
    BEST = np.load(O / "ref_best.npz")["env"].astype(np.float64) if (O / "ref_best.npz").exists() else None
    out = open(O / f"ppca_{TAG}.txt", "w"); t0 = time.time()
    def say(*a):
        s = " ".join(str(x) for x in a); print(f"[{time.time() - t0:5.0f}s] " + s, flush=True); out.write(s + "\n"); out.flush()
    say(f"=== {TAG}: stack {STACK}, K {K}, {IT} EM iterations, band {LO}-{HI} A")
    stack = W / ("inputs/sim_homog_nuposes/particles.128.mrcs" if NULL else "inputs/particles.128.mrcs")
    bp = FastBP(str(stack), str(W / "nu_consensus/poses.pkl"), str(W / "inputs/ctf.pkl")); N = bp.data.N; D = bp.D; d2 = D // 2; dev = bp.dev
    band = bp.band_mask(LO, HI); coords = bp.lat.coords[bp.mask][band]; Mb = int(band.sum())
    shell = (bp.lat.freqs2d[bp.mask][band].norm(dim=1) * D).round().long(); say("band pixels", Mb, "| Fourier radius", int(shell.min()), "-", int(shell.max()))
    V = np.load(A / SCORES)[:, 0]; rank = np.argsort(np.argsort(V)) / N; NBm = 20; pbin = np.minimum((rank * NBm).astype(int), NBm - 1)
    maps = [bp.hvol(np.where(pbin == k)[0]) for k in range(NBm)]; say("PC1-bin maps done")
    # ---- whitened residuals of ALL particles
    R = torch.empty((N, Mb), device=dev); Cw = torch.empty((N, Mb), device=dev); ROT = torch.empty((N, 3, 3), device=dev)
    with torch.no_grad():
        for s in range(0, N, 2000):
            ii = np.arange(s, min(N, s + 2000)); ff, rot, c = bp._batch(ii); ff = ff[:, band]; c = c[:, band]; P = torch.empty_like(ff); lb = pbin[ii]
            for m in np.unique(lb):
                rr = np.flatnonzero(lb == m); P[rr] = projK(maps[m][None], rot[rr], coords, D)[:, 0]
            pred = c * P; al = (ff * pred).sum(1, keepdim=True) / pred.pow(2).sum(1, keepdim=True).clamp(min=1e-20)
            R[s:s + len(ii)] = ff - al * pred; Cw[s:s + len(ii)] = al * c; ROT[s:s + len(ii)] = rot
        nsh = int(shell.max()) + 1; var = torch.zeros(nsh, device=dev, dtype=torch.float64)
        for s in range(0, N, 20000): var.index_add_(0, shell, R[s:s + 20000].pow(2).sum(0).double())
        sd = (var / (N * torch.bincount(shell, minlength=nsh).clamp(min=1))).sqrt().float()[shell]; R /= sd; Cw /= sd
    if NOISENULL:
        gN = torch.Generator(device=dev).manual_seed(1234)
        for s in range(0, N, 20000): R[s:s + 20000] = torch.randn(R[s:s + 20000].shape, generator=gN, device=dev)
        NULL = True; say("NOISE-ONLY control: residuals replaced by unit Gaussian noise")
    say(f"switches: fold seed {FOLDSEED}, scores {SCORES}, noise null {NOISENULL}")
    del maps; torch.cuda.empty_cache(); say(f"whitened residuals stored; mean |a CTF|^2/noise per pixel {float(Cw[::50].pow(2).mean()):.3g}")
    msk = torch.tensor(gaussian_filter(ENV.astype(np.float32), 1.5), device=dev); vox = band_voxels(D, float(shell.max()) + 1.0, dev)
    perm = np.random.default_rng(FOLDSEED).permutation(N); F = [np.sort(perm[:N // 2]), np.sort(perm[N // 2:])]; Ft = [torch.as_tensor(f, device=dev) for f in F]
    QQ, TT = [], []
    for f in (0, 1):
        say(f"fold {f}: EM on {len(F[f])} particles"); Q, tau2 = fit(R, Cw, ROT, Ft[f], K, IT, coords, D, vox, msk, seed=f, say=say); QQ.append(Q); TT.append(tau2)
    # ---- fold agreement (label-free) and alignment of fold 1's axes to fold 0's
    M = (QQ[0].reshape(K, -1).double() @ QQ[1].reshape(K, -1).double().T); Ua, S, Vh = torch.linalg.svd(M); Rot = (Ua @ Vh).float()
    say("tau2 fold 0:", " ".join(f"{x:.4g}" for x in TT[0].tolist())); say("tau2 fold 1:", " ".join(f"{x:.4g}" for x in TT[1].tolist()))
    say("fold-vs-fold subspace cosines (principal angles):", " ".join(f"{x:.3f}" for x in S.tolist()))
    say("fold-vs-fold |cc| of same-rank axes:", " ".join(f"{abs(float(M[k, k])):.3f}" for k in range(K)))
    for j in (2, 4, K):
        say(f"   leading-{j} subspaces: cosines", " ".join(f"{x:.3f}" for x in torch.linalg.svdvals(M[:j, :j]).tolist()))
    z = np.zeros((N, K), np.float32)
    zh, _ = estep(R, Cw, ROT, Ft[1], QQ[0], TT[0], coords, D); z[F[1]] = (zh * TT[0].sqrt()).cpu().numpy()
    zh, _ = estep(R, Cw, ROT, Ft[0], QQ[1], TT[1], coords, D); z[F[0]] = ((zh * TT[1].sqrt()) @ Rot.T).cpu().numpy()
    del R, Cw; torch.cuda.empty_cache()
    real = lambda u: np.asarray(fft.ihtn_center(u[0:-1, 0:-1, 0:-1].cpu())).astype(np.float32)
    E0 = np.array([real(q) for q in QQ[0]]); E = np.array([lowpass(e, bp.apix, 10.0)[ENV].astype(np.float64) for e in E0])
    np.save(O / f"ppca_{TAG}_z.npy", z); np.save(O / f"ppca_{TAG}_eigvol_fold0.npy", E0)
    say("out-of-fold coefficient variance per axis:", " ".join(f"{x:.4g}" for x in z.var(0)), "| fold 0 vs fold 1 particles:", " ".join(f"{z[F[1], k].var() / max(z[F[0], k].var(), 1e-30):.2f}" for k in range(K)))
    if NOEVAL:
        say("no-evaluation run: coefficients saved; done"); sys.exit()
    cc = lambda x, y: float(np.corrcoef(x, y)[0, 1])
    cap = lambda v, B: float(v @ (B.T @ np.linalg.lstsq(B.T, v, rcond=None)[0]) / (v @ v))
    say("EVAL share of each class direction (10 A, envelope) lying inside the fold-0 subspace [leading 2 / 4 / all axes]:")
    for nm, v in (("P7-own", own), ("P6-P8", uu), ("AltNBD1 - consensus", alt)):
        say(f"   {nm:20s} {cap(v, E[:2]):.3f} / {cap(v, E[:4]):.3f} / {cap(v, E):.3f}   per axis cc: " + " ".join(f"{cc(E[k], v):+.2f}" for k in range(K)))
    if NULL:
        say("null run: done"); sys.exit()
    # ---- EVALUATION ONLY below
    m3 = (lab >= 0) & (lab <= 2); auc = lambda y, x: max(roc_auc_score(y, x), 1 - roc_auc_score(y, x))
    say("EVAL per axis: corr with PC1 | AUC P7-vs-rest of P6/P7/P8 | AUC AltNBD1-vs-all | AUC other(decoy/absent)-vs-all")
    for k in range(K):
        say(f"   axis {k + 1}: {cc(z[:, k], V):+.3f} | {auc(lab[m3] == 1, z[m3, k]):.3f} | {auc(lab == 3, z[:, k]):.3f} | {auc(lab == -1, z[:, k]):.3f}")
    cv = StratifiedKFold(5, shuffle=True, random_state=0); zs = (z - z.mean(0)) / z.std(0); Vs = (V - V.mean()) / V.std(); prob = {}
    for nm, X in (("residual coefficients only", zs), ("PC1 only", Vs[:, None]), ("PC1 + residual coefficients", np.hstack([Vs[:, None], zs]))):
        p = cross_val_predict(LogisticRegression(max_iter=300), X[m3], lab[m3], cv=cv, method="predict_proba"); pr = p.argmax(1); y = lab[m3]; prob[nm] = p[:, 1]
        say(f"EVAL supervised ceiling, {nm:28s}: acc {np.mean(pr == y):.3f} recall P6/P7/P8 {np.mean(pr[y == 0] == 0):.2f}/{np.mean(pr[y == 1] == 1):.2f}/{np.mean(pr[y == 2] == 2):.2f}"
            f"  P7 AUC {roc_auc_score(y == 1, p[:, 1]):.3f}   (cryoDRGN latents: P7 AUC 0.70, recall 0.17-0.25)")

    def score(name, idx):
        h = [lowpass(x, bp.apix, 10.0)[ENV].astype(np.float64) for x in bp.halves(np.asarray(idx), seed=5)]
        dd = [dist(h[0], h[1], RA[j], RB[j]) / d86 for j in range(3)]; t = (((h[0] + h[1]) / 2 - (RA[2] + RB[2]) / 2) @ uu) / un
        car = 0.5 * ((h[0] - (RA[2] + t * (RA[0] - RA[2]))) @ ownB + (h[1] - (RB[2] + t * (RB[0] - RB[2]))) @ ownA) / own2
        bs = "" if BEST is None else "  best-map P6/P7/P8 " + "/".join(f"{float(((h[0] - BEST[j]) * (h[1] - BEST[j])).sum() / ((BEST[2] - BEST[0]) ** 2).sum()):.4f}" for j in range(3))
        say(f"  {name:50s} n={len(idx):6d}  P6 {dd[0]:.4f}  P7 {dd[1]:.4f}  P8 {dd[2]:.4f}  t={t:5.2f}  own-P7 {car:5.2f}  [P7 share {np.mean(lab[idx] == 1):.2f}]{bs}")
        return dd[1]

    say("VOLUME TEST (fixed poses, 10 A, envelope; distance^2 / d(P8,P6); consensus P7 0.060, mid-PC1 window 0.041, pure P7 0.003)")
    mid = np.where((rank >= 0.30) & (rank < 0.70))[0]; score("mid-PC1 window (reference)", mid); res = []
    for k in range(K):
        o = mid[np.argsort(z[mid, k])]; q = len(mid) * 30 // 100
        res.append((score(f"axis {k + 1} LOW 30 % of the mid-PC1 window", o[:q]), f"axis {k + 1} low")); res.append((score(f"axis {k + 1} HIGH 30 % of the mid-PC1 window", o[-q:]), f"axis {k + 1} high"))
    res.sort(); say("  -> best single-axis splits by P7 distance:", [(round(a, 4), b) for a, b in res[:4]])
    for Kc in (3, 4, 5, 6):
        X = np.hstack([Vs[:, None], zs[:, :4]]); g = GaussianMixture(Kc, covariance_type="full", random_state=0, n_init=2).fit(X[::4]); cl = g.predict(X)
        for c in np.argsort(g.means_[:, 0]):
            ix = np.where(cl == c)[0]
            if len(ix) > 3000: score(f"GMM K={Kc} on [PC1, axes 1-4], cluster at PC1 {g.means_[c, 0]:+.2f}", ix)
    for Kc in (2, 3):
        g = GaussianMixture(Kc, covariance_type="full", random_state=0, n_init=3).fit(zs[mid][:, :4]); cl = g.predict(zs[mid][:, :4])
        for c in range(Kc):
            if (cl == c).sum() > 3000: score(f"GMM K={Kc} on axes 1-4 inside the mid window, cluster {c}", mid[cl == c])
    say("  -- ORACLE ceilings (use labels; not a method)")
    ids = np.where(m3)[0]
    for nm in ("residual coefficients only", "PC1 + residual coefficients"):
        o = ids[np.argsort(-prob[nm])]
        for n in (20000, 40000): score(f"ORACLE supervised top {n // 1000}k, {nm}", o[:n])
    say("done")
