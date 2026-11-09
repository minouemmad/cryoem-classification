#!/usr/bin/env python
"""How well does a trained cryoDRGN VAE reproduce the particle images when (A) z is perturbed with the PE fixed,
or (B) the PE frequencies are perturbed with z fixed? Per-particle fit and class averages along latent bins.
Usage: vae_perturb_vs_data.py --pe-run <pe_tests dir> -o OUT [--bins 150 --per-bin 200]
"""
import argparse
import json
import math
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

import pe_tests as T
from cryodrgn import fft

VIEW_DEG = 3.5


def fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = math.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def rotate_imgs(img, theta, c):
    """rotate B x H x W images by theta (rad) about pixel (c, c); bilinear, zeros outside."""
    B, H, W = img.shape
    ys, xs = torch.meshgrid(torch.arange(H, device=img.device, dtype=img.dtype),
                            torch.arange(W, device=img.device, dtype=img.dtype), indexing="ij")
    x, y = xs - c, ys - c
    ct, st = torch.cos(theta).view(B, 1, 1), torch.sin(theta).view(B, 1, 1)
    xi, yi = ct * x - st * y + c, st * x + ct * y + c
    grid = torch.stack([xi / (W - 1) * 2 - 1, yi / (H - 1) * 2 - 1], -1)
    return F.grid_sample(img.unsqueeze(1), grid, mode="bilinear", padding_mode="zeros", align_corners=True).squeeze(1)


def frc(a, b, shell, nsh):
    """FRC by shell between two real images (B x D x D) via FFT."""
    fa, fb = torch.fft.fftshift(torch.fft.fft2(a), dim=(-2, -1)), torch.fft.fftshift(torch.fft.fft2(b), dim=(-2, -1))
    num = torch.zeros(len(a), nsh, device=a.device, dtype=torch.float64)
    pa, pb = torch.zeros_like(num), torch.zeros_like(num)
    s = shell.expand(len(a), -1)
    num.scatter_add_(1, s, (fa * fb.conj()).real.flatten(1).double())
    pa.scatter_add_(1, s, (fa.abs() ** 2).flatten(1).double())
    pb.scatter_add_(1, s, (fb.abs() ** 2).flatten(1).double())
    return num / (pa * pb).sqrt().clamp(min=1e-30)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pe-run", required=True, help="pe_tests.py output dir (for its args: VAE run, particles, poses, ctf)")
    p.add_argument("-o", required=True)
    p.add_argument("--bins", type=int, default=150)
    p.add_argument("--per-bin", type=int, default=200)
    p.add_argument("--n-dirs", type=int, default=1500)
    # 2026-11-07: run on another model / pose set, and choose the noise levels (both in SD units of the perturbed input:
    # z noise in per-dimension SDs of z; frequency noise in SDs of the encoding frequencies B, i.e. 0.01 = the old "1 %")
    p.add_argument("--run", default=None, help="cryoDRGN run dir (default: the VAE of --pe-run)")
    p.add_argument("--poses", default=None, help="poses.pkl (default: the one of --pe-run)")
    p.add_argument("--z-sd", type=float, nargs="*", default=[0.25, 0.5, 1.0, 2.0])
    p.add_argument("--pe-sd", type=float, nargs="*", default=[0.01, 0.03, 0.1, 0.3])
    p.add_argument("--base-only", action="store_true")
    a = p.parse_args()
    import os
    os.makedirs(a.o, exist_ok=True)
    args = SimpleNamespace(**json.load(open(f"{a.pe_run}/results.json"))["args"])
    if a.run:
        args.run = a.run
    if a.poses:
        args.poses = a.poses
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    ctx = T.Ctx(args, dev)
    N, D = ctx.data.N, ctx.D           # D = 129 lattice; real images are (D-1) x (D-1)
    Dr = D - 1
    rot_all = ctx.pt.rots.detach().cpu().numpy() if hasattr(ctx.pt, "rots") else None
    if rot_all is None:
        rot_all = torch.cat([ctx.pt.get_pose(torch.arange(s, min(s + 4096, N), device=dev))[0].cpu()
                             for s in range(0, N, 4096)]).numpy()
    rot_all = rot_all.reshape(N, 3, 3)

    # ---- view bins: particles whose viewing direction (row 2 of R; slice plane = rows 0,1) is within VIEW_DEG of a grid dir
    v = rot_all[:, 2, :]
    dirs = fib_sphere(a.n_dirs)
    best, bestc = np.zeros(N, int), np.full(N, -2.0)
    for s in range(0, N, 20000):
        c = v[s:s + 20000] @ dirs.T
        best[s:s + 20000], bestc[s:s + 20000] = c.argmax(1), c.max(1)
    ok = bestc > np.cos(np.deg2rad(VIEW_DEG))
    cnt = np.bincount(best[ok], minlength=a.n_dirs)
    order = np.argsort(-cnt)[:a.bins]
    rng = np.random.default_rng(0)
    members, bin_of = [], {}
    for b_i, d in enumerate(order):
        m = np.where(ok & (best == d))[0]
        m = np.sort(rng.choice(m, min(a.per_bin, len(m)), replace=False))
        members.append(m)
        for i in m:
            bin_of[int(i)] = b_i
    sel = np.sort(np.concatenate(members))
    print(f"{len(order)} view bins, particles per bin {min(map(len, members))}-{max(map(len, members))}, total {len(sel)}", flush=True)
    # in-plane angle of each particle relative to its bin's reference particle (first member)
    theta = np.zeros(N)
    for m in members:
        R0 = rot_all[m[0]]
        for i in m:
            Ri = rot_all[i]
            theta[i] = math.atan2(R0[0] @ Ri[1], R0[0] @ Ri[0])   # Rz(theta) @ Ri has row 0 closest to R0[0]

    # ---- variants
    ref = ctx.R0.decoder.rand_freqs.data.detach().clone()
    zstd = ctx.z.std(0)
    zmean = ctx.z.mean(0)
    g = torch.Generator(device="cpu").manual_seed(31)
    eps = torch.randn(ctx.z.shape, generator=g).to(dev)
    perm = torch.from_numpy(np.random.default_rng(7).permutation(N)).to(dev)
    noise = torch.randn(ref.shape, generator=torch.Generator().manual_seed(777)).to(dev)
    newdraw = (torch.randn(ref.shape, generator=torch.Generator().manual_seed(12345)) * 0.5).to(dev)
    Z = {"base": None}
    PEV = {}
    if not a.base_only:
        for d in a.z_sd:
            Z[f"z_noise_{d:g}sd"] = ("noise", d)
        Z["z_other_particle"] = ("other", None)
        Z["z_dataset_mean"] = ("mean", None)
        PEV = {f"pe_noise_{d:g}": ref + d * ref.std() * noise for d in a.pe_sd}
        PEV["pe_new_draw"] = newdraw
    names = list(Z) + list(PEV)

    def zfor(kind, ind):
        z = ctx.z[ind]
        if kind is None:
            return z
        if kind[0] == "noise":
            return z + kind[1] * zstd * eps[ind]
        if kind[0] == "other":
            return ctx.z[perm[ind]]
        return zmean.expand_as(z)

    def real(v):
        img = torch.zeros(len(v), D * D, device=v.device)
        img[:, ctx.mask] = v
        img = img.view(-1, D, D)[:, :-1, :-1]
        return torch.stack([fft.iht2_center(im) for im in img])

    # ---- rotation sign self-check with noise-free predictions
    chk = torch.from_numpy(sel[:8]).to(dev)
    with torch.no_grad():
        Rr = ctx.pt.get_pose(chk)[0]
        th = torch.full((8,), 0.4, device=dev)
        Rz = torch.zeros(8, 3, 3, device=dev)
        Rz[:, 0, 0] = Rz[:, 1, 1] = torch.cos(th); Rz[:, 0, 1] = torch.sin(th); Rz[:, 1, 0] = -torch.sin(th); Rz[:, 2, 2] = 1
        s1 = real(T.slices(ctx.R0.decoder, ctx.coords, Rr, ctx.z[chk]))
        s2 = real(T.slices(ctx.R0.decoder, ctx.coords, Rz @ Rr, ctx.z[chk]))
        cc = {sg: float(F.cosine_similarity(rotate_imgs(s1, sg * th, Dr / 2).flatten(1), s2.flatten(1)).mean()) for sg in (1, -1)}
    SIGN = max(cc, key=cc.get)
    print("rotation self-check (corr with directly rendered in-plane-rotated slice):", cc, "-> sign", SIGN, flush=True)

    # ---- accumulate
    nb = len(members)
    acc = {n: torch.zeros(nb, Dr, Dr, device=dev, dtype=torch.float64) for n in names}
    dat = torch.zeros(2, nb, Dr, Dr, device=dev, dtype=torch.float64)
    nper = torch.zeros(2, nb, device=dev)
    mse = {n: np.zeros(len(sel)) for n in names}
    mse0 = np.zeros(len(sel))
    pos = {int(i): k for k, i in enumerate(sel)}
    half = {int(i): (j % 2) for m in members for j, i in enumerate(m)}
    t0 = time.time()
    done = 0
    with torch.no_grad():
        for mb in T.loader(ctx, sel, 128, False):
            ind, y, rot, c = ctx.batch(mb)
            il = ind.cpu().numpy()
            k = np.array([pos[int(i)] for i in il])
            bidx = torch.tensor([bin_of[int(i)] for i in il], device=dev)
            hidx = torch.tensor([half[int(i)] for i in il], device=dev)
            thb = torch.tensor(SIGN * theta[il], device=dev, dtype=torch.float32)
            sg = torch.sign(c)
            yr = rotate_imgs(real(y * sg), thb, Dr / 2).double()
            for h in (0, 1):
                mh = hidx == h
                dat[h].index_add_(0, bidx[mh], yr[mh])
                nper[h].index_add_(0, bidx[mh], torch.ones(int(mh.sum()), device=dev))
            mse0[k] = (y ** 2).mean(1).cpu().numpy()
            for n in names:
                dec = ctx.R0.decoder
                if n in PEV:
                    dec.rand_freqs.data.copy_(PEV[n])
                s = T.slices(dec, ctx.coords, rot, zfor(Z.get(n), ind))
                dec.rand_freqs.data.copy_(ref)
                mse[n][k] = ((s * c - y) ** 2).mean(1).cpu().numpy()
                acc[n].index_add_(0, bidx, rotate_imgs(real(s * c * sg), thb, Dr / 2).double())
            done += len(ind)
            if done % (128 * 40) < 128:
                print(f"{done}/{len(sel)} {time.time() - t0:.0f}s", flush=True)

    # ---- metrics
    nsh = Dr // 2 + 1
    yy, xx = torch.meshgrid(torch.arange(Dr, device=dev), torch.arange(Dr, device=dev), indexing="ij")
    rr = ((yy - Dr // 2) ** 2 + (xx - Dr // 2) ** 2).float().sqrt().round().long().clamp(max=nsh - 1)
    shell = rr.flatten().unsqueeze(0)
    circ = (rr < Dr // 2 - 8).double()
    ntot = (nper[0] + nper[1]).view(-1, 1, 1)
    davg = (dat[0] + dat[1]) / ntot
    fh = frc((dat[0] / nper[0].view(-1, 1, 1)).float(), (dat[1] / nper[1].view(-1, 1, 1)).float(), shell, nsh)
    fhalf = fh.mean(0).clamp(min=0)
    cref = (2 * fhalf / (1 + fhalf)).sqrt()
    w = ntot.view(-1)
    out = dict(run=str(args.run), poses=str(args.poses), mse0=float(mse0.mean()),
               n_particles=int(len(sel)), n_bins=nb, view_deg=VIEW_DEG, per_bin=a.per_bin, rotation_selfcheck=cc,
               data_half_frc=fhalf.cpu().tolist(), c_ref=cref.cpu().tolist(), models={})
    base_mse = mse["base"]
    for n in names:
        pavg = acc[n] / ntot
        f = frc(pavg.float(), davg.float(), shell, nsh)
        f_mean = (f * w.view(-1, 1)).sum(0) / w.sum()
        a_c, b_c = (pavg - (pavg * circ).sum((1, 2), keepdim=True) / circ.sum()) * circ, \
                   (davg - (davg * circ).sum((1, 2), keepdim=True) / circ.sum()) * circ
        rcls = ((a_c * b_c).sum((1, 2)) / (a_c.norm(dim=(1, 2)) * b_c.norm(dim=(1, 2)))).cpu().numpy()
        dlt = mse[n] - base_mse
        out["models"][n] = dict(
            expl=float(1 - mse[n].mean() / mse0.mean()),
            dmse_mean=float(dlt.mean()), dmse_ci95=float(1.96 * dlt.std() / np.sqrt(len(dlt))),
            dexpl_pct_points=float(-dlt.mean() / mse0.mean() * 100),
            base_better_frac=float((dlt > 0).mean()),
            class_frc=f_mean.cpu().tolist(), class_r_median=float(np.median(rcls)), class_r_mean=float(rcls.mean()))
        print(f"  {n:20s} expl={out['models'][n]['expl'] * 100:6.3f} %  d(expl)={out['models'][n]['dexpl_pct_points']:+.4f} pts "
              f"base better {out['models'][n]['base_better_frac'] * 100:5.1f} %  class r={np.median(rcls):.3f}", flush=True)
    json.dump(out, open(f"{a.o}/results.json", "w"), indent=1)
    ex = order[:0]
    ex_bins = list(range(min(8, nb)))
    np.savez(f"{a.o}/class_averages.npz", bins=np.array(ex_bins), counts=ntot.view(-1)[ex_bins].cpu().numpy(),
             data=davg[ex_bins].cpu().numpy(),
             data_half0=(dat[0] / nper[0].view(-1, 1, 1))[ex_bins].cpu().numpy(),
             **{n: (acc[n] / ntot)[ex_bins].cpu().numpy() for n in names})
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
