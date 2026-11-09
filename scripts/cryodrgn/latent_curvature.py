#!/usr/bin/env python
"""Intrinsic geometry of a cryoDRGN latent through its decoder: pullback metric G = J^T J of the decoded
volume (speed, effective dimension, volume element), sectional curvature, and local data dimension / density, at
random and tail particles. Writes points.json, summary.json (per leaf with --leaves) and geometry.png.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mrcfile
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors

from cryo_utils import load_latents


class Decoder64:
    def __init__(self, run: Path, epoch: int, box: int, device: str):
        from cryodrgn import config
        from cryodrgn.models import HetOnlyVAE
        cfg = config.load(str(run / "config.yaml"))
        self.norm = [float(x) for x in cfg["dataset_args"]["norm"]]
        model, lat = HetOnlyVAE.load(cfg, str(run / f"weights.{epoch}.pkl"), device=device)
        self.dec = model.decoder
        self.dec.eval()
        D = cfg["lattice_args"]["D"]
        self.box = box
        self.ext = lat.extent * (box / (D - 1))
        self.coords = lat.get_downsample_coords(box + 1)
        self.apix = float(cfg["dataset_args"].get("Apix", 0) or 0)
        self.n = 0

    def __call__(self, z: np.ndarray) -> np.ndarray:
        self.n += 1
        return self.dec.eval_volume(self.coords, self.box + 1, self.ext, self.norm, z.astype(np.float32)).cpu().numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--epoch", type=int, default=50)
    p.add_argument("--mask64", type=Path, required=True)
    p.add_argument("--apix64", type=float, required=True, help="voxel size of the 64 px decode (A)")
    p.add_argument("--leaves", type=Path, help="recon_tree leaves.npy for per-leaf summaries")
    p.add_argument("--points", type=int, default=300)
    p.add_argument("--tail", type=int, default=20)
    p.add_argument("--h", type=float, default=0.25)
    p.add_argument("--h2", type=float, default=0.1)
    p.add_argument("--h2-points", type=int, default=40)
    p.add_argument("--random-planes", type=int, default=3)
    p.add_argument("--fwhm", type=float, default=8.0)
    p.add_argument("--knn", type=int, default=200)
    p.add_argument("--ref-volumes", type=int, default=200, help="random particles decoded for the mean and r")
    p.add_argument("--box", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-o", "--outdir", type=Path, required=True)
    a = p.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()

    z = load_latents(a.run / f"z.{a.epoch}.pkl").astype(np.float64)
    N, d = z.shape
    sd = z.std(0)
    mask = mrcfile.read(a.mask64).astype(np.float32)
    sel = mask > 0.05
    wm = mask[sel]
    dec = Decoder64(a.run, a.epoch, a.box, "cuda")
    sig = a.fwhm / 2.355 / a.apix64

    def feat(zz):
        return gaussian_filter(dec(zz), sig)[sel] * wm

    # reference scale: mean decoded volume and RMS distance r of decoded volumes from it
    ref = np.stack([feat(z[i]) for i in rng.choice(N, a.ref_volumes, replace=False)])
    mu = ref.mean(0)
    r = float(np.sqrt(((ref - mu) ** 2).sum(1).mean()))
    unit = lambda v: v / r      # volume vectors in units of r

    # latent PCs, sample points
    zc = z - z.mean(0)
    _, _, Pt = np.linalg.svd(zc[rng.choice(N, 50000, replace=False)], full_matrices=False)
    pts = list(rng.choice(N, a.points, replace=False))
    proj = zc @ Pt[:2].T
    for k in range(2):
        o = np.argsort(proj[:, k])
        pts += list(o[: a.tail // 4]) + list(o[-a.tail // 4:])
    pts = np.array(pts)
    kind = ["random"] * a.points + ["tail"] * (len(pts) - a.points)
    nn = NearestNeighbors(n_neighbors=a.knn).fit(z[rng.choice(N, min(N, 100000), replace=False)])
    leaves = np.load(a.leaves) if a.leaves else None

    def geometry(z0, h_sd, planes):
        """Jacobian, metric, curvature of the given latent planes (pairs of unit vectors in z units)."""
        h = h_sd * sd                                     # per-dimension step
        f0 = unit(feat(z0))
        J = np.empty((len(f0), d))
        plus, minus = [], []
        for j in range(d):
            e = np.zeros(d); e[j] = h[j]
            fp, fm = unit(feat(z0 + e)), unit(feat(z0 - e))
            J[:, j] = (fp - fm) / (2 * h[j])
        G = J.T @ J
        if callable(planes):
            planes = planes(G)
        lam = np.clip(np.linalg.eigvalsh(G)[::-1], 0, None)
        Q, _ = np.linalg.qr(J)
        perp = lambda v: v - Q @ (Q.T @ v)
        out = dict(speed=float(np.sqrt(lam.sum())), eff_dim=float(lam.sum() ** 2 / (lam ** 2).sum()),
                   logvol=float(0.5 * np.log(np.clip(lam, 1e-30, None)).sum()), eig=lam.tolist(), K={})
        hs = float(h_sd)
        for name, (u, v) in planes.items():
            # step along u, v measured in latent sd units (scale-free directions)
            du, dv = u * sd * hs, v * sd * hs
            fu_p, fu_m, fv_p, fv_m = (unit(feat(z0 + x)) for x in (du, -du, dv, -dv))
            fpp, fpm, fmp, fmm = (unit(feat(z0 + x)) for x in (du + dv, du - dv, -du + dv, -du - dv))
            Va, Vb = (fu_p - fu_m) / (2 * hs), (fv_p - fv_m) / (2 * hs)
            Vaa, Vbb = (fu_p - 2 * f0 + fu_m) / hs ** 2, (fv_p - 2 * f0 + fv_m) / hs ** 2
            Vab = (fpp - fpm - fmp + fmm) / (4 * hs ** 2)
            IIaa, IIbb, IIab = perp(Vaa), perp(Vbb), perp(Vab)
            den = (Va @ Va) * (Vb @ Vb) - (Va @ Vb) ** 2
            out["K"][name] = float((IIaa @ IIbb - IIab @ IIab) / den) if den > 0 else None
        return out, G

    rows = []
    for i, (row, kd) in enumerate(zip(pts, kind)):
        z0 = z[row]
        def planes_of(G):
            S = np.diag(sd)
            _, U = np.linalg.eigh(S @ G @ S)               # metric in latent-sd units
            U = U[:, ::-1]
            pl = {"metric_top2": (U[:, 0], U[:, 1])}
            q0, q1 = Pt[0] / sd, Pt[1] / sd
            pl["latent_pc12"] = (q0 / np.linalg.norm(q0), q1 / np.linalg.norm(q1))
            for k in range(a.random_planes):
                q, _ = np.linalg.qr(rng.standard_normal((d, 2)))
                pl[f"random{k}"] = (q[:, 0], q[:, 1])
            planes_of.last = pl
            return pl
        g, G = geometry(z0, a.h, planes_of)
        planes = planes_of.last
        if i < a.h2_points:
            g3, _ = geometry(z0, a.h2, {"metric_top2": planes["metric_top2"]})
            g["K_h2_metric_top2"] = g3["K"]["metric_top2"]
        dist, nb = nn.kneighbors(z0[None])
        loc = nn._fit_X[nb[0]]
        ev = np.clip(np.linalg.eigvalsh(np.cov(loc.T)), 0, None)
        g.update(row=int(row), kind=kd, z=z0.tolist(), local_dim=float(ev.sum() ** 2 / (ev ** 2).sum()),
                 density=float(1 / dist[0, -1]), leaf=int(leaves[row]) if leaves is not None else None,
                 pc=(zc[row] @ Pt[:2].T).tolist())
        rows.append(g)
        if i % 20 == 0:
            print(f"{i}/{len(pts)} {time.time() - t0:.0f}s K_top2 {g['K'].get('metric_top2')} speed {g['speed']:.3f}", flush=True)
        if i % 50 == 0:
            (a.outdir / "points.json").write_text(json.dumps(dict(r=r, rows=rows)))

    (a.outdir / "points.json").write_text(json.dumps(dict(r=r, rows=rows)))
    K = lambda key: np.array([x["K"][key] for x in rows if x["K"].get(key) is not None])
    rnd = np.array([np.mean([x["K"][k] for k in x["K"] if k.startswith("random")]) for x in rows])
    summ = dict(run=str(a.run), epoch=a.epoch, r=r, h=a.h, n_points=len(rows), decodes=dec.n,
                eff_dim_median=float(np.median([x["eff_dim"] for x in rows])),
                speed_median=float(np.median([x["speed"] for x in rows])),
                local_dim_median=float(np.median([x["local_dim"] for x in rows])),
                K={k: dict(median=float(np.median(K(k))), p10=float(np.percentile(K(k), 10)), p90=float(np.percentile(K(k), 90)),
                           frac_negative=float((K(k) < 0).mean())) for k in ("metric_top2", "latent_pc12")},
                K_random_mean=dict(median=float(np.median(rnd)), frac_negative=float((rnd < 0).mean())))
    h2 = [(x["K"]["metric_top2"], x["K_h2_metric_top2"]) for x in rows if "K_h2_metric_top2" in x]
    if h2:
        h2 = np.array(h2)
        summ["scale_check"] = dict(h=a.h, h2=a.h2, spearman=float(__import__("scipy.stats").stats.spearmanr(h2[:, 0], h2[:, 1])[0]),
                                   median_ratio=float(np.median(h2[:, 1] / h2[:, 0])))
    if leaves is not None:
        summ["by_leaf"] = {}
        for L in sorted({x["leaf"] for x in rows}):
            R = [x for x in rows if x["leaf"] == L]
            summ["by_leaf"][str(L)] = dict(n=len(R), K_top2_median=float(np.median([x["K"]["metric_top2"] for x in R])),
                                           speed_median=float(np.median([x["speed"] for x in R])),
                                           eff_dim_median=float(np.median([x["eff_dim"] for x in R])))
    # correlations of curvature with speed / density / local dimension
    from scipy.stats import spearmanr
    kt = np.array([x["K"]["metric_top2"] for x in rows])
    summ["spearman_K_top2"] = {k: float(spearmanr(kt, [x[k] for x in rows])[0]) for k in ("speed", "density", "local_dim", "eff_dim")}
    (a.outdir / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    zs = zc[rng.choice(N, 20000, replace=False)] @ Pt[:2].T
    pc = np.array([x["pc"] for x in rows])
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
    for axx, (lab, val, cm) in zip(ax, [("sectional curvature K r^2 (top-2 metric plane)", kt, "RdBu_r"),
                                        ("speed |dV/dz|", [x["speed"] for x in rows], "viridis"),
                                        ("effective decoder dimension", [x["eff_dim"] for x in rows], "viridis"),
                                        ("local data dimension", [x["local_dim"] for x in rows], "viridis")]):
        axx.scatter(zs[:, 0], zs[:, 1], s=1, c="0.85", rasterized=True)
        val = np.asarray(val, float)
        if cm == "RdBu_r":
            lim = np.percentile(np.abs(val), 90)
            sc = axx.scatter(pc[:, 0], pc[:, 1], c=val, cmap=cm, vmin=-lim, vmax=lim, s=14, edgecolor="k", linewidth=0.2)
        else:
            sc = axx.scatter(pc[:, 0], pc[:, 1], c=val, cmap=cm, s=14, edgecolor="k", linewidth=0.2)
        plt.colorbar(sc, ax=axx, fraction=0.046)
        axx.set_title(lab, fontsize=9); axx.set_xlabel("latent PC1"); axx.set_ylabel("latent PC2")
    fig.tight_layout()
    fig.savefig(a.outdir / "geometry.png", dpi=130)


if __name__ == "__main__":
    main()
