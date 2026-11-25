#!/usr/bin/env python
"""Reconstruction-guided divisive tree over a cryoDRGN run's decoded centres.

At each node: take the dominant axis (volume PC1 of the node's decoded centres, or latent PC1 with --axis latent),
score particles along it, propose K = --ks segments by 1-D k-means, and accept a split only if it passes held-out
reconstruction tests (fastbp half maps): adjacent segments must differ reproducibly beyond a random-partition null, and
the children must fit held-out images better than size-matched random subsets (--accept cv). Recurse until no split
passes. Writes leaves.npy (one label per particle, cryoDRGN order), the tree JSON and the log.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mrcfile
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors

from cryo_utils import kmeans1d, load_class_set, load_latents, read_volume, volume_features
from fastbp import FastBP, fsc_band, fsc_curve, lowpass, resolution


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def soft_mask(vol, apix, lp=15.0, thr_frac=0.15, dilate_A=6.0, edge_A=6.0):
    v = lowpass(vol, apix, lp)
    core = v > thr_frac * v.max()
    core = binary_dilation(core, iterations=max(1, int(round(dilate_A / apix))))
    d = distance_transform_edt(~core) * apix
    return np.where(core, 1.0, np.clip(0.5 * (1 + np.cos(np.pi * d / edge_A)), 0, None) * (d < edge_A)).astype(np.float32)


class Axis:
    """Weighted PC1 (or PC `which`) of centre features via the n_c x n_c Gram matrix; bootstrap in that space."""

    def __init__(self, F, which=1):
        self.F = F.astype(np.float64)
        self.K = self.F @ self.F.T
        self.which = which

    def pc1(self, w):
        w = w / w.sum()
        a = self.K @ w
        c = w @ a
        G0 = self.K - a[:, None] - a[None, :] + c
        s = np.sqrt(w)
        G = s[:, None] * G0 * s[None, :]
        lam, u = eigsh(G, k=self.which, which="LA")             # ascending: [0] is the which-th largest
        su = s * u[:, 0]
        return su, float(lam[0]), w, G0

    def direction(self, w):
        su, lam, w, _ = self.pc1(w)
        mu = w @ self.F
        v = self.F.T @ su - mu * su.sum()
        return v / np.linalg.norm(v), mu, lam

    def bootstrap(self, w, B, rng):
        su0, lam0, w0, G0 = self.pc1(w)
        cos = []
        n = len(w)
        for _ in range(B):
            m = rng.multinomial(n, np.full(n, 1 / n)).astype(float)
            if (m * w).sum() == 0:
                continue
            su, lam, wb, _ = self.pc1(w * m)
            # v0 . vb in centre space: both are (F - 1 mu^T)^T (s u); cross term via K
            mu0, mub = w0 @ self.F, wb @ self.F
            Fa, Fb = self.F.T @ su0 - mu0 * su0.sum(), self.F.T @ su - mub * su.sum()
            cos.append(abs(Fa @ Fb) / (np.linalg.norm(Fa) * np.linalg.norm(Fb)))
        return np.array(cos)


class Curve:
    """Weighted principal curve (Hastie-Stuetzle) through the node's centres, in their top --curve-dims weighted PCs.

    Starts from PC1 and alternates (a) curve = kernel-weighted running mean of the centres ordered by their current
    coordinate (bandwidth = --curve-span x the weighted spread of the coordinate) and (b) re-projection of every centre
    onto the curve (nearest point of the densely sampled polyline); the coordinate is arc length. With a large span it
    reduces to PC1. The bootstrap analogue of the axis |cos| test is |Spearman| between the centres' coordinates on
    the full-data curve and on a curve refit to bootstrap-resampled centre weights.
    """

    def __init__(self, F, dims, span, iters=15, grid=400, keep=50, start=1):
        # work in the top `keep` PCs of the centre maps (Gram-matrix SVD): exact distances up to the discarded tail,
        # and every fit / bootstrap refit is then cheap
        F = F.astype(np.float64)
        Fc = F - F.mean(0)
        lam, U = np.linalg.eigh(Fc @ Fc.T)
        top = np.argsort(lam)[::-1][:keep]
        self.F = U[:, top] * np.sqrt(np.clip(lam[top], 0, None))
        self.dims, self.span, self.iters, self.grid = dims, span, iters, grid
        self.start = start                      # curve through PCs start..start+dims-1, initialised at PC `start`

    def _pcs(self, w):
        w = w / w.sum()
        mu = w @ self.F
        Xc = (self.F - mu) * np.sqrt(w)[:, None]
        _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        sl = slice(self.start - 1, self.start - 1 + self.dims)
        return (self.F - mu) @ Vt[sl].T, (s ** 2 / (s ** 2).sum())[sl]

    def fit(self, w):
        Y, ev = self._pcs(w)
        w = w / w.sum()
        lam = Y[:, 0].copy()
        for _ in range(self.iters):
            sd = np.sqrt(w @ (lam - w @ lam) ** 2)
            g = np.linspace(lam.min(), lam.max(), self.grid)
            K = np.exp(-0.5 * ((g[:, None] - lam[None]) / (self.span * sd)) ** 2) * w[None]
            P = (K @ Y) / K.sum(1, keepdims=True)                    # curve sampled on the grid
            seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
            d2 = ((Y ** 2).sum(1)[:, None] + (P ** 2).sum(1)[None] - 2 * Y @ P.T)
            new = seg[np.argmin(d2, 1)]
            if np.corrcoef(new, lam)[0, 1] < 0:
                new = seg[-1] - new
            done = np.allclose(new, lam, atol=1e-6 * max(seg[-1], 1e-12))
            lam = new
            if done:
                break
        res = (w * np.clip(np.min(d2, 1), 0, None)).sum() / (w * ((Y - w @ Y) ** 2).sum(1)).sum()
        tot_dims = ev.sum()
        return lam, dict(curve_var_frac=float((1 - res) * tot_dims), pc1_var_frac=float(ev[0]), curve_length=float(seg[-1]))

    def bootstrap(self, w, B, rng, lam0):
        from scipy.stats import spearmanr
        out = []
        n = len(w)
        for _ in range(B):
            m = rng.multinomial(n, np.full(n, 1 / n)).astype(float)
            if (m * w).sum() == 0:
                continue
            lb, _ = self.fit(w * m + 1e-3 * w)    # every centre keeps a tiny weight so all get a coordinate
            out.append(abs(spearmanr(lam0, lb)[0]))
        return np.array(out)


def choose_k(C, proj, ks=(1, 2, 4, 8, 16, 32)):
    kmax = min(max(ks) + 1, len(C))
    idx = NearestNeighbors(n_neighbors=kmax).fit(C).kneighbors(C, return_distance=False)[:, 1:]
    err = {k: float(np.mean((proj[idx[:, :k]].mean(1) - proj) ** 2)) for k in ks if k < kmax}
    return min(err, key=err.get), err


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--decoded", type=Path, required=True)
    p.add_argument("--mask64", type=Path, required=True, help="feature mask (decoded box)")
    p.add_argument("--inputs", type=Path, required=True, help="dir with particles.128/256.mrcs, poses.pkl, ctf.pkl")
    p.add_argument("--particles-cs", type=Path, required=True)
    p.add_argument("--class-set", nargs="+", required=True)
    p.add_argument("--smooth-fwhm", type=float, default=8.0)
    p.add_argument("--ks", type=int, nargs="+", default=[4, 3, 2], help="numbers of 1-D k-means segments proposed per node")
    p.add_argument("--boots", type=int, default=200)
    p.add_argument("--axis-p05", type=float, default=0.8)
    p.add_argument("--nulls", type=int, default=5)
    p.add_argument("--controls", type=int, default=3)
    p.add_argument("--min-gain", type=float, default=0.006, help="min size-weighted mean FSC-band gain over random (3 x the 0.002 scatter of random-subset reconstructions)")
    p.add_argument("--z", type=float, default=5.0)
    p.add_argument("--lp", type=float, default=8.0)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--min-n", type=int, default=3000)
    p.add_argument("--min-centres", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--accept", choices=("gain", "prefer-split", "cv"), default="gain",
                   help="gain (v3): split only if children are more homogeneous than random by --min-gain. prefer-split: "
                        "split if test 1 passes, children are not measurably LESS homogeneous than random (mean gain > "
                        "-max(--gain-tol, --gain-tol-sd x control sd)) and the partition is stable under bootstrap of the "
                        "centres (median ARI >= --min-ari); the largest such K is taken")
    p.add_argument("--cv-band", type=float, nargs=2, default=[30.0, 6.0], help="--accept cv: resolution band (A) of the image fit")
    p.add_argument("--cv-nulls", type=int, default=3, help="--accept cv: permuted-label null draws")
    p.add_argument("--cv-z", type=float, default=5.0, help="--accept cv: excess must exceed this many null sd")
    p.add_argument("--cv-min-rel", type=float, default=0.0, help="--accept cv: and this fraction of the mean consensus score")
    p.add_argument("--cv-rule", choices=("predictive", "excess", "corrected"), default="predictive",
                   help="--accept cv K selection. predictive (default): plain cross-validation - the children's own maps must "
                        "explain held-out images better than the consensus (raw mean own-minus-consensus > --cv-z x paired sem, "
                        "and excess over the permuted null > --cv-z x null sd); the K with the largest raw statistic wins, so "
                        "the reconstruction-noise cost of smaller maps limits K. excess: the K with the largest null-subtracted "
                        "excess wins (rises monotonically with K on a continuum - over-splits). corrected: the fold maps carry the "
                        "noise of half-size reconstructions; the noise cost c (= -mean of the permuted-label statistic) is also "
                        "measured at quarter size (group AND consensus maps from half of the training fold, one permutation), "
                        "c(size) ~ size^-b gives b = log2(c_q / c_h), and the cost of full-size group maps is extrapolated, "
                        "c_full = c_h^2 / c_q (falls back to c_h / 2 if c_q <= c_h). score = excess - c_full; pass if score > "
                        "--cv-z x sqrt(paired sem^2 + null sd^2) and excess > --cv-z x null sd; the largest score wins")
    p.add_argument("--cv-test-n", type=int, default=40000, help="--accept cv: held-out images scored per fold (subsample)")
    p.add_argument("--gain-tol", type=float, default=0.002)
    p.add_argument("--gain-tol-sd", type=float, default=1.0)
    p.add_argument("--min-ari", type=float, default=0.8)
    p.add_argument("--stab-boots", type=int, default=50)
    p.add_argument("--stab-n", type=int, default=20000, help="particles subsampled for the bootstrap partition ARI")
    p.add_argument("--axis", choices=("pc1", "curve", "latent"), default="pc1",
                   help="pc1: straight weighted PC1 of the centre maps. curve: weighted principal curve through the "
                        "centres (see Curve); reproducibility = bootstrap |Spearman| p05 >= --axis-p05")
    p.add_argument("--k-rule", choices=("best", "incremental"), default="best",
                   help="best: among passing K (--ks) take the one with the largest gain/score. incremental (2026-11-07): "
                        "try K = 2, 3, 4, ... in ascending order and stop at the first K that fails or does not raise the "
                        "gain/score over the previous K; the last K that improved is taken (gain and cv rules)")
    p.add_argument("--root-pc", type=int, default=1, help="principal component used for the root split (deeper nodes "
                                                                  "use PC1); e.g. 2 to look for structure orthogonal to PC1")
    p.add_argument("--axis-candidates", type=int, nargs="+", default=None,
                   help="test these straight PCs at every node (e.g. 1 2 3): each must pass --axis-p05, the axis + K "
                        "with the largest accepted gain is used (gain rule only)")
    p.add_argument("--gain-mask", type=Path, default=None,
                   help="256 px map on the consensus grid (e.g. molmap of a fitted model) defining the test-2 FSC mask")
    p.add_argument("--curve-dims", type=int, default=10, help="weighted PCs of the centre maps the curve lives in")
    p.add_argument("--curve-span", type=float, default=0.3, help="smoother bandwidth in units of the coordinate sd")
    p.add_argument("--curve-boots", type=int, default=50)
    p.add_argument("-o", "--outdir", type=Path, required=True)
    a = p.parse_args()
    if a.axis_candidates and (a.accept == "prefer-split" or a.axis != "pc1"):
        p.error("--axis-candidates is implemented for --accept gain / cv with straight axes only")
    if a.axis == "curve" and a.accept == "prefer-split":
        p.error("--axis curve is implemented for --accept gain / cv only")
    a.outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    S0 = 1000 * a.seed   # --seed also offsets the half-split / null / control seeds of tests 1, 2, 2' (seed 0 = previous runs)

    meta = json.loads((a.decoded / "sample.json").read_text())
    run = Path(meta["run"])
    z = load_latents(run / f"z.{meta['epoch']}.pkl")
    C = np.loadtxt(a.decoded / "centers.txt")
    paths = [a.decoded / "volumes" / f"vol_{i + 1:03d}.mrc" for i in range(meta["samples"])]
    mask64 = read_volume(a.mask64, meta["box"], meta["apix"])
    F = volume_features(paths, mask64, meta["box"], meta["apix"], a.smooth_fwhm, "correlation", 32)
    ax = Axis(F, which=a.root_pc)
    nearest = NearestNeighbors(n_neighbors=1).fit(C).kneighbors(z, return_distance=False)[:, 0]
    N = len(z)

    bp_lo = FastBP(str(a.inputs / "particles.128.mrcs"), str(a.inputs / "poses.pkl"), str(a.inputs / "ctf.pkl"))
    bp_hi = None if a.accept == "cv" else FastBP(str(a.inputs / "particles.256.mrcs"), str(a.inputs / "poses.pkl"), str(a.inputs / "ctf.pkl"))
    if bp_hi is None:
        bp_hi = bp_lo      # cv: no 256 px maps; leaf resolutions are reported at 128 px
    allidx = np.arange(N)
    log("consensus maps for masks")
    m_lo = soft_mask(sum(bp_lo.halves(allidx)), bp_lo.apix)
    m_hi = soft_mask(sum(bp_hi.halves(allidx)), bp_hi.apix)
    if a.gain_mask is not None:
        # protein-only FSC mask for test 2 (e.g. a molmap of a fitted model on the 256 px consensus grid), so that
        # micelle / detergent-belt differences cannot make a split look more homogeneous
        gm = mrcfile.read(a.gain_mask).astype(np.float32)
        assert gm.shape == m_hi.shape, (gm.shape, m_hi.shape)
        m_hi = soft_mask(gm, bp_hi.apix, lp=6.0, thr_frac=0.1, dilate_A=3.0, edge_A=6.0)
        log("test-2 FSC mask from", a.gain_mask, "(%.1f%% of the box)" % (100 * (m_hi > 0.5).mean()))
    np.save(a.outdir / "mask128.npy", m_lo)
    np.save(a.outdir / "mask256.npy", m_hi)
    sel = m_lo > 0.5
    n_lo, n_hi = m_lo.shape[0], m_hi.shape[0]

    cache_lo, cache_hi = {}, {}

    def halves_lo(idx, seed):
        key = (hash(np.sort(idx).tobytes()), seed)
        if key not in cache_lo:
            cache_lo[key] = [lowpass(h, bp_lo.apix, a.lp) for h in bp_lo.halves(idx, seed)]
        return cache_lo[key]

    def res_hi(idx, seed=0):
        """(band-mean FSC 10-3 A, 0.143 resolution) of masked half maps at 256 px."""
        key = (hash(np.sort(idx).tobytes()), seed)
        if key not in cache_hi:
            h = bp_hi.halves(idx, seed)
            c = fsc_curve(h[0], h[1], m_hi)
            cache_hi[key] = (fsc_band(c, n_hi, bp_hi.apix), resolution(c, n_hi, bp_hi.apix))
        return cache_hi[key]

    def diff_r(i_idx, j_idx, seed):
        Ai, Bi = halves_lo(i_idx, seed)
        Aj, Bj = halves_lo(j_idx, seed)
        return float(np.corrcoef((Ai - Aj)[sel], (Bi - Bj)[sel])[0, 1])

    def null_r(i_idx, j_idx):
        u = np.concatenate([i_idx, j_idx])
        out = []
        for s in range(a.nulls):
            perm = np.random.default_rng(10_000 + S0 + s).permutation(u)
            out.append(diff_r(perm[: len(i_idx)], perm[len(i_idx):], seed=S0 + s + 1))
        return np.array(out)

    cv_band = bp_lo.band_mask(*a.cv_band) if a.accept == "cv" else None

    def cv_test(nid, K, rows, parts):
        """Test 2' (--accept cv): held-out image fit of own-child maps vs consensus, against permuted-label nulls."""
        lab = np.full(N, -1)
        for j, q in enumerate(parts):
            lab[q] = j
        frng = np.random.default_rng(40_000 + a.seed + len(rows))
        fold = frng.random(len(rows)) < 0.5
        d_real, d_null, s_cons_all, own_lab, acc_hits, d_quarter = [], [[] for _ in range(a.cv_nulls)], [], [], [], []
        for f in (0, 1):
            train, test = rows[fold == bool(f)], rows[fold != bool(f)]
            if len(test) > a.cv_test_n:
                test = np.sort(frng.choice(test, a.cv_test_n, replace=False))
            Hc = bp_lo.hvol(train)
            sc = bp_lo.fit(test, [Hc], np.zeros(len(test), int), cv_band)
            del Hc
            Hs = [bp_lo.hvol(train[lab[train] == j]) for j in range(K)]
            so = bp_lo.fit(test, Hs, lab[test], cv_band)
            allj = np.stack([bp_lo.fit(test, Hs, np.full(len(test), j), cv_band) for j in range(K)], 1)
            acc_hits.append(allj.argmax(1) == lab[test])
            del Hs
            d_real.append(so - sc); s_cons_all.append(sc); own_lab.append(lab[test])
            for t in range(a.cv_nulls):
                pl = np.full(N, -1)
                pl[rows] = np.random.default_rng(50_000 + S0 + 97 * t + f).permutation(lab[rows])
                Hn = [bp_lo.hvol(train[pl[train] == j]) for j in range(K)]
                d_null[t].append(bp_lo.fit(test, Hn, pl[test], cv_band) - sc)
                del Hn
                if a.cv_rule == "corrected" and t == 0:
                    # same permuted-label statistic at quarter size: group and consensus maps from half of the fold
                    tq = train[np.random.default_rng(60_000 + S0 + f).random(len(train)) < 0.5]
                    Hq = bp_lo.hvol(tq)
                    scq = bp_lo.fit(test, [Hq], np.zeros(len(test), int), cv_band)
                    del Hq
                    Hnq = [bp_lo.hvol(tq[pl[tq] == j]) for j in range(K)]
                    d_quarter.append(bp_lo.fit(test, Hnq, pl[test], cv_band) - scq)
                    del Hnq
        d_real = np.concatenate(d_real); sc = np.concatenate(s_cons_all); ol = np.concatenate(own_lab)
        dn = np.stack([np.concatenate(x) for x in d_null])            # (nulls, n_test)
        null_means = dn.mean(1)
        excess = float(d_real.mean() - null_means.mean())
        null_sd = float(null_means.std(ddof=1)) if a.cv_nulls > 1 else 0.0
        need = max(a.cv_z * null_sd, a.cv_min_rel * float(sc.mean()))
        sem = float(d_real.std() / np.sqrt(len(d_real)))
        raw = float(d_real.mean())
        per_child = [dict(n_test=int((ol == j).sum()), excess=float(d_real[ol == j].mean() - dn[:, ol == j].mean()), raw=float(d_real[ol == j].mean()))
                     for j in range(K)]
        acc = float(np.concatenate(acc_hits).mean())
        chance = float((np.bincount(ol, minlength=K) / len(ol)).max())
        res = dict(cv_excess=excess, cv_excess_rel=excess / float(sc.mean()), cv_null_sd=null_sd, cv_required=need,
                   cv_stat=float(d_real.mean()), cv_null_means=null_means.tolist(), cv_paired_sem=float(d_real.std() / np.sqrt(len(d_real))),
                   cv_children=per_child, cv_assign_acc=acc, cv_assign_chance=chance, cv_consensus_score=float(sc.mean()))
        log(nid, "K", K, "cv raw own-cons %.3f (sem %.3f) | excess %.3f (%.2f%%) null sd %.3f required %.3f | assign acc %.3f (chance %.3f) | children %s" %
            (raw, sem, excess, 100 * res["cv_excess_rel"], null_sd, need, acc, chance, [round(c["excess"], 2) for c in per_child]))
        if a.cv_rule == "corrected":
            c_h = -float(null_means.mean())
            c_q = -float(np.concatenate(d_quarter).mean())
            if c_h > 0 and c_q > c_h:
                b, c_full = float(np.log2(c_q / c_h)), c_h * c_h / c_q
            else:
                b, c_full = float("nan"), c_h / 2
            score = excess - c_full
            need_c = a.cv_z * float(np.sqrt(sem ** 2 + null_sd ** 2))
            ok = excess > need and score > need_c
            res.update(cv_rule="corrected", cv_score=score, cv_cost_half=c_h, cv_cost_quarter=c_q, cv_cost_exponent=b,
                       cv_cost_full=c_full, cv_required_corrected=need_c)
            log(nid, "K", K, "corrected: cost half %.3f quarter %.3f -> exponent %.2f, full-size cost %.3f, score %.3f (required %.3f)%s" %
                (c_h, c_q, b, c_full, score, need_c, " PASS" if ok else ""))
            return ok, score, res
        if a.cv_rule == "predictive":
            ok = excess > need and raw > a.cv_z * sem
            res.update(cv_rule="predictive", cv_score=raw)
            return ok, raw, res
        res.update(cv_rule="excess", cv_score=excess)
        return excess > need, excess, res

    def run_proposals(nid, rows, s, boot_s=None, sidx=None):
        """1-D k-means proposals along particle scores s (K from --ks, largest first) and the two tests.
        Returns (accepted (K, thr, side, parts) or None, its mean gain, proposal records)."""
        rec = dict(proposals=[])
        accepted, best_gain = None, None
        rec["proposals"] = []
        inc = a.k_rule == "incremental" and a.accept != "prefer-split"
        for K in sorted(set(a.ks), reverse=not inc):
            thr = kmeans1d(s, K)
            side = np.searchsorted(thr, s)
            parts = [rows[side == j] for j in range(K)]
            prop = dict(K=K, sizes=[len(q) for q in parts], pairs=[], children=[])
            if min(len(q) for q in parts) < a.min_n:
                prop["result"] = "child below floor"
                rec["proposals"].append(prop)
                if inc:
                    break
                continue
            ok = True
            for j in range(K - 1):
                r = diff_r(parts[j], parts[j + 1], seed=S0)
                nr = null_r(parts[j], parts[j + 1])
                thr_r = nr.mean() + a.z * max(nr.std(), 0.02)
                prop["pairs"].append(dict(pair=[j, j + 1], r=r, null_mean=float(nr.mean()),
                                          null_sd=float(nr.std()), threshold=float(thr_r), passed=bool(r > thr_r)))
                log(nid, "K", K, "pair", j, round(r, 3), "null", round(nr.mean(), 3), round(nr.std(), 3))
                ok &= r > thr_r
            if not ok:
                prop["result"] = "children not distinguishable"
                rec["proposals"].append(prop)
                if inc:
                    break
                continue
            if a.accept == "cv":
                passed, excess, cres = cv_test(nid, K, rows, parts)
                prop.update(cres, result="cv gain" if passed else "no cross-validated fit gain")
                rec["proposals"].append(prop)
                if passed and (best_gain is None or excess > best_gain):
                    accepted, best_gain = (K, thr, side, parts), excess
                elif inc:
                    prop["result"] += " (incremental stop: no gain over K-1)" if passed else ""
                    log(nid, "K", K, "incremental stop; keeping K", accepted[0] if accepted else None)
                    break
                continue
            gains, sds = [], []
            for j, q in enumerate(parts):
                fb, rc = res_hi(q, seed=S0)
                ctl = [res_hi(np.random.default_rng(20_000 + S0 + 37 * c + j).choice(rows, len(q), replace=False), seed=S0 + c + 1)
                       for c in range(a.controls)]
                cb = np.array([c_[0] for c_ in ctl])
                gains.append(fb - cb.mean()); sds.append(cb.std(ddof=1))
                prop["children"].append(dict(fsc_band=fb, res=rc, ctrl_fsc_band=cb.tolist(),
                                             ctrl_res=[c_[1] for c_ in ctl], gain=float(fb - cb.mean())))
                log(nid, "K", K, "child", j, "fscband", round(fb, 3), "ctrl", cb.round(3), "gain", round(fb - cb.mean(), 3))
            wq = np.array([len(q) for q in parts], float)
            mean_gain = float((wq * np.array(gains)).sum() / wq.sum())
            noise = float(np.sqrt(np.mean(np.square(sds))))
            if a.accept == "gain":
                need = max(a.min_gain, 3 * noise)
                passed = mean_gain > need
                prop.update(mean_gain=mean_gain, control_sd=noise, required_gain=need,
                            result="gain" if passed else "no homogeneity gain")
                log(nid, "K", K, "mean gain", round(mean_gain, 4), "required", round(need, 4))
                rec["proposals"].append(prop)
                if passed and (best_gain is None or mean_gain > best_gain):
                    accepted, best_gain = (K, thr, side, parts), mean_gain
                elif inc:
                    log(nid, "K", K, "incremental stop; keeping K", accepted[0] if accepted else None)
                    break
                continue
            # prefer-split: not measurably less homogeneous than random, and a bootstrap-stable partition
            need = -max(a.gain_tol, a.gain_tol_sd * noise)
            aris = np.array([adjusted_rand_score(side[sidx], np.searchsorted(kmeans1d(sb, K), sb)) for sb in boot_s])
            ari = float(np.median(aris))
            passed = mean_gain > need and ari >= a.min_ari
            prop.update(mean_gain=mean_gain, control_sd=noise, required_gain=need, stab_ari_median=ari,
                        stab_ari_p05=float(np.percentile(aris, 5)),
                        result="split" if passed else ("less homogeneous than random" if mean_gain <= need
                                                       else "partition not bootstrap-stable"))
            log(nid, "K", K, "mean gain", round(mean_gain, 4), "floor", round(need, 4), "stab ARI", round(ari, 3))
            rec["proposals"].append(prop)
            if passed:                               # K runs largest first: the first pass is the largest K
                accepted, best_gain = (K, thr, side, parts), mean_gain
                break
        return accepted, best_gain, rec["proposals"]

    nodes, leaves = [], np.full(N, -1)
    queue = [("r", allidx, np.arange(len(C)), 0)]
    while queue:
        nid, rows, cidx, depth = queue.pop(0)
        rec = dict(node=nid, depth=depth, n=int(len(rows)), n_centres=int(len(cidx)))
        log("node", nid, rec)
        reason = None
        if depth >= a.max_depth:
            reason = "max depth"
        elif len(rows) < 2 * a.min_n or len(cidx) < 2 * a.min_centres:
            reason = "below computational floor"
        if reason is None and a.axis_candidates:
            # several straight PCs per node: each must pass the bootstrap gate, then gets the full proposals; the axis
            # and K with the largest accepted mean gain win
            w = np.bincount(nearest[rows], minlength=len(C))[cidx] + 1.0
            sub = Axis(F[cidx]) if len(cidx) < len(C) else Axis(F)
            rec["candidates"], best = [], None
            for pc in a.axis_candidates:
                sub.which = pc
                v, mu, lam = sub.direction(w)
                cos = sub.bootstrap(w, a.boots, rng)
                cand = dict(pc=pc, eigenvalue=lam, axis_cos_median=float(np.median(cos)), axis_cos_p05=float(np.percentile(cos, 5)))
                log(nid, "PC%d bootstrap median |cos| %.3f, p05 %.3f" % (pc, cand["axis_cos_median"], cand["axis_cos_p05"]))
                if cand["axis_cos_p05"] < a.axis_p05:
                    cand["result"] = "axis not reproducible"
                    rec["candidates"].append(cand)
                    continue
                cp = (F[cidx] - mu) @ v
                k, _ = choose_k(C[cidx], cp)
                s = cp[NearestNeighbors(n_neighbors=k).fit(C[cidx]).kneighbors(z[rows], return_distance=False)].mean(1)
                acc, bg, props = run_proposals(f"{nid}[PC{pc}]", rows, s)
                cand.update(score_k=k, proposals=props, accepted_K=acc[0] if acc else None, best_gain=bg,
                            result="split" if acc else "no proposal passed")
                rec["candidates"].append(cand)
                if acc is not None and (best is None or bg > best[1]):
                    best = (acc, bg, pc, cp)
            if best is None:
                reason = "no candidate axis passed the reconstruction tests"
            else:
                accepted, cproj = best[0], best[3]
                rec.update(chosen_pc=best[2], accepted_K=accepted[0])
                log(nid, "chosen PC%d K=%d gain %.4f" % (best[2], accepted[0], best[1]))
        elif reason is None:
            w = np.bincount(nearest[rows], minlength=len(C))[cidx] + 1.0
            if a.axis == "latent":
                # latent-space axis (2026-11-07): PC1 of the node particles' z (unweighted PCA); particle score = projection
                # on it; bootstrap = PC1 of resampled particles (20k subsample per draw)
                zr = z[rows]
                lmu = zr.mean(0)
                lev, lvec = np.linalg.eigh(np.cov((zr - lmu).T))
                lv = lvec[:, -1]
                rec.update(latent_pc1_var_frac=float(lev[-1] / lev.sum()))
                cos = []
                for _ in range(min(a.boots, 50)):
                    b = zr[rng.choice(len(zr), min(20000, len(zr)), replace=True)]
                    eb = np.linalg.eigh(np.cov((b - b.mean(0)).T))[1][:, -1]
                    cos.append(abs(float(eb @ lv)))
                cos = np.array(cos)
                log(nid, "latent PC1 explains %.3f of z variance" % rec["latent_pc1_var_frac"])
            elif a.axis == "curve":
                cv = Curve(F[cidx], a.curve_dims, a.curve_span)
                cproj, cinfo = cv.fit(w)
                cos = cv.bootstrap(w, a.curve_boots, rng, cproj)
                rec.update(cinfo)
                log(nid, "curve explains %.3f of variance (PC1 %.3f), length %.3g" %
                    (cinfo["curve_var_frac"], cinfo["pc1_var_frac"], cinfo["curve_length"]))
            else:
                sub = Axis(F[cidx]) if len(cidx) < len(C) else ax
                v, mu, lam = sub.direction(w)
                cos = sub.bootstrap(w, a.boots, rng)
            log(nid, "axis bootstrap median |cos| %.3f, p05 %.3f" % (np.median(cos), np.percentile(cos, 5)))
            rec.update(axis_cos_median=float(np.median(cos)), axis_cos_p05=float(np.percentile(cos, 5)))
            if rec["axis_cos_p05"] < a.axis_p05:
                reason = "axis not reproducible (bootstrap p05 |cos| < %.2f)" % a.axis_p05
        if reason is None and not a.axis_candidates:
            if a.axis == "pc1":
                cproj = (F[cidx] - mu) @ v
            if a.axis == "latent":
                s = (z[rows] - lmu) @ lv                 # particle scores directly on the latent axis
                cproj = (C[cidx] - lmu) @ lv             # centres (latent k-means centres) only follow their side
            else:
                k, err = choose_k(C[cidx], cproj)
                rec.update(score_k=k, score_k_loo_mse={str(kk): e for kk, e in err.items()})
                idx = NearestNeighbors(n_neighbors=k).fit(C[cidx]).kneighbors(z[rows], return_distance=False)
                s = cproj[idx].mean(1)
            if a.accept == "prefer-split":
                # test 3 inputs: bootstrap-resampled centres -> axis, centre projections and particle scores
                from sklearn.metrics import adjusted_rand_score
                sidx = rng.choice(len(rows), min(a.stab_n, len(rows)), replace=False)
                boot_s = []
                for _ in range(a.stab_boots):
                    mb = rng.multinomial(len(cidx), np.full(len(cidx), 1 / len(cidx))).astype(float)
                    if (mb * w).sum() == 0:
                        continue
                    vb, mub, _ = sub.direction(w * mb)
                    if vb @ v < 0:
                        vb = -vb
                    boot_s.append(((F[cidx] - mub) @ vb)[idx[sidx]].mean(1))
            accepted, best_gain, props = run_proposals(nid, rows, s, boot_s if a.accept == "prefer-split" else None,
                                                       sidx if a.accept == "prefer-split" else None)
            rec["proposals"] = props
            if accepted is not None:
                rec["accepted_K"] = accepted[0]
            if accepted is None:
                reason = "no proposal passed the reconstruction tests"
        if reason is not None:
            leaves[rows] = len([n for n in nodes if n.get("leaf")])
            rec.update(leaf=True, reason=reason, leaf_label=int(leaves[rows[0]]))
            nodes.append(rec)
            log("leaf", nid, reason)
            continue
        K, thr, side, parts = accepted
        cside = np.searchsorted(thr, cproj)
        rec.update(leaf=False, K=K, thresholds=thr.tolist())
        nodes.append(rec)
        for j, q in enumerate(parts):
            queue.append((nid + str(j), q, cidx[cside == j], depth + 1))
        (a.outdir / "nodes.json").write_text(json.dumps(nodes, indent=1))

    # evaluation only
    cs = np.load(a.particles_cs)
    _, names, cls = load_class_set(a.class_set, cs["uid"])
    from sklearn.metrics import adjusted_mutual_info_score
    for rec in nodes:
        if rec.get("leaf"):
            m = leaves == rec["leaf_label"]
            rec["class_frac"] = {nm: float((cls[m] == j).mean()) for j, nm in enumerate(names)}
            rec["fsc_band256"], rec["res256"] = res_hi(np.flatnonzero(m))
    keep = cls >= 0
    out = dict(run=str(run), epoch=meta["epoch"], params={k: str(v) for k, v in vars(a).items()},
               n_leaves=int(leaves.max() + 1), ami_classes=float(adjusted_mutual_info_score(cls[keep], leaves[keep])),
               nodes=nodes)
    (a.outdir / "nodes.json").write_text(json.dumps(out, indent=1))
    np.save(a.outdir / "leaves.npy", leaves)
    log("done", out["n_leaves"], "leaves, AMI", round(out["ami_classes"], 3))


if __name__ == "__main__":
    main()
