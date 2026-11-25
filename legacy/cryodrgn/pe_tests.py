#!/usr/bin/env python
"""2-D slice tests of the decoder positional encoding (PE) with the latent held fixed: the trained VAE (R0)
vs altered PE frequencies (zero-shot) and decoders retrained with other PEs, scored on held-out particles.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import BatchSampler, DataLoader, SubsetRandomSampler

from cryodrgn import ctf, dataset, fft
from cryodrgn.lattice import Lattice
from cryodrgn.models import FTPositionalDecoder, HetOnlyVAE, ResidLinearMLP
from cryodrgn.pose import PoseTracker


class RawKDecoder(FTPositionalDecoder):
    """FTPositionalDecoder whose 'encoding' is the identity: MLP input = (kx, ky, kz, z)."""

    def __init__(self, zdim, D, nlayers, hidden):
        super().__init__(3 + zdim, D, nlayers, hidden, torch.nn.ReLU, enc_type="gaussian", feat_sigma=0.5)
        del self.rand_freqs
        self.rand_freqs = None
        self.decoder = ResidLinearMLP(3 + zdim, nlayers, hidden, 2, torch.nn.ReLU)

    def positional_encoding_geom(self, coords):
        return coords


class StochPEDecoder(FTPositionalDecoder):
    """gaussian-PE decoder with stochasticity in the encoding during training only.
      resample : B ~ N(0, sigma^2) redrawn at every forward pass (truly stochastic PE); eval uses self.rand_freqs
                 (one fixed draw) -- see EnsembleDec for the draw-averaged prediction.
      reparam  : B = mu + exp(log_s) * eps, eps ~ N(0, I); mu and log_s learned, no KL; eval uses mu.
      kjitter  : k -> k + N(0, jit^2) (in lattice units) before the fixed PE; eval uses k.
      ipe      : fixed PE, features scaled by exp(-1/2 (D2 tau |b_j|)^2) = E[feature] under k-jitter of std tau; tau fixed
                 (= jit) or learned (log_tau). Deterministic, same in train and eval."""

    def __init__(self, zdim, D, mode, ref_freqs, sigma=0.5, s0=0.05, jit=0.5 / 128, learn_tau=False):
        super().__init__(3 + zdim, D, 3, 1024, torch.nn.ReLU, enc_type="gaussian", feat_sigma=sigma)
        self.mode, self.sigma, self.jit = mode, sigma, jit
        self.rand_freqs.data.copy_(ref_freqs)
        if mode == "ipe" and learn_tau:
            self.log_tau = torch.nn.Parameter(torch.tensor(float(np.log(jit))))
        if mode == "reparam":
            self.mu = torch.nn.Parameter(ref_freqs.clone())
            self.log_s = torch.nn.Parameter(torch.full_like(ref_freqs, float(np.log(s0 * sigma))))

    def tau(self):
        return self.log_tau.exp() if hasattr(self, "log_tau") else torch.tensor(self.jit, device=self.rand_freqs.device)

    def current_B(self):
        if self.mode == "resample" and self.training:
            return torch.randn_like(self.rand_freqs) * self.sigma
        if self.mode == "reparam":
            return self.mu + (torch.exp(self.log_s) * torch.randn_like(self.mu) if self.training else 0.0)
        return self.rand_freqs

    def random_fourier_encoding(self, coords):
        if self.mode == "kjitter" and self.training:
            xyz = (coords[..., 0:3] + self.jit * torch.randn_like(coords[..., 0:3])).clamp(-0.5, 0.5)
            coords = torch.cat([xyz, coords[..., 3:]], -1)
        freqs = self.current_B().view(*[1] * (len(coords.shape) - 1), -1, 3) * self.D2
        k = (coords[..., None, 0:3] * freqs).sum(-1)
        x = torch.cat([torch.sin(k), torch.cos(k)], -1)
        if self.mode == "ipe":
            att = torch.exp(-0.5 * (self.D2 * self.tau()) ** 2 * (self.rand_freqs ** 2).sum(-1))
            x = x * torch.cat([att, att])
        x = x.view(*coords.shape[:-1], self.in_dim - self.zdim)
        return torch.cat([x, coords[..., 3:]], -1)


class EnsembleDec(torch.nn.Module):
    """prediction averaged over M fixed PE draws of a 'resample' decoder (the best a stochastic PE can do)."""

    def __init__(self, dec, M=8, seed=999):
        super().__init__()
        self.dec = dec
        g = torch.Generator().manual_seed(seed)
        self.draws = [torch.randn(dec.rand_freqs.shape, generator=g) * dec.sigma for _ in range(M)]

    def forward(self, lat):
        keep = self.dec.rand_freqs.data.clone()
        out = 0
        for B in self.draws:
            self.dec.rand_freqs.data.copy_(B.to(keep.device))
            out = out + self.dec(lat)
        self.dec.rand_freqs.data.copy_(keep)
        return out / len(self.draws)


def make_decoder(kind, zdim, D, ref_freqs, seed):
    torch.manual_seed(seed)
    if kind == "raw_k":
        return RawKDecoder(zdim, D, 3, 1024)
    if kind in ("pe_resample", "pe_reparam", "k_jitter"):
        mode = {"pe_resample": "resample", "pe_reparam": "reparam", "k_jitter": "kjitter"}[kind]
        f = ref_freqs if kind != "pe_resample" else torch.randn(ref_freqs.shape, generator=torch.Generator().manual_seed(4242)) * 0.5
        return StochPEDecoder(zdim, D, mode, f)
    if kind in ("ipe_jit", "ipe_2px", "ipe_learn"):
        return StochPEDecoder(zdim, D, "ipe", ref_freqs, jit=(2.0 if kind == "ipe_2px" else 0.5) / 128, learn_tau=kind == "ipe_learn")
    if kind == "geom_lowf":
        return FTPositionalDecoder(3 + zdim, D, 3, 1024, torch.nn.ReLU, enc_type="geom_lowf")
    d = FTPositionalDecoder(3 + zdim, D, 3, 1024, torch.nn.ReLU, enc_type="gaussian", feat_sigma=0.5)
    f = {"pe_orig": ref_freqs, "pe_s025": ref_freqs * 0.5, "pe_s1": ref_freqs * 2.0}.get(kind)
    if kind == "pe_new":
        f = torch.randn(ref_freqs.shape, generator=torch.Generator().manual_seed(12345)) * 0.5
    d.rand_freqs.data.copy_(f)
    return d


def slices(dec, coords, rot, z):
    """pre-CTF slice values on the masked lattice: B x Npix"""
    x = coords @ rot  # B x N x 3
    zz = z[:, None, :].expand(-1, x.shape[1], -1)
    return dec(torch.cat([x, zz], -1)).view(x.shape[0], -1)


class Ctx:
    def __init__(self, a, dev):
        cfg = a.run + "/config.yaml"
        self.R0, lat = HetOnlyVAE.load(cfg, f"{a.run}/weights.{a.epoch}.pkl", device=dev)
        self.R0.eval()
        self.data = dataset.ImageDataset(mrcfile=a.particles, lazy=True, invert_data=True, window=True,
                                         window_r=0.85, max_threads=16, device="cpu")
        N, D = self.data.N, self.data.D
        self.D = D
        self.lat = Lattice(D, extent=0.5, device=dev)
        self.mask = self.lat.get_circular_mask(D // 2)
        self.coords = self.lat.coords[self.mask] / self.lat.extent / 2
        self.pt = PoseTracker.load(a.poses, N, D, None, None, device=dev)
        self.ctfp = torch.tensor(ctf.load_ctf_for_training(D - 1, a.ctf), device=dev)
        with open(f"{a.run}/z.{a.epoch}.pkl", "rb") as f:
            self.z = torch.tensor(pickle.load(f), dtype=torch.float32, device=dev)
        r = (self.lat.freqs2d[self.mask].norm(dim=1) * (D - 1)).round().long()
        self.shell = r.clamp(max=D // 2)
        self.nsh = D // 2 + 1
        self.dev = dev

    def batch(self, mb):
        ind = mb["index"].to(self.dev)
        y = mb["y"].to(self.dev)
        B = len(ind)
        rot, tran = self.pt.get_pose(ind)
        y = self.lat.translate_ht(y.view(B, -1), tran.unsqueeze(1)).view(B, -1)[:, self.mask]
        cp = self.ctfp[ind]
        fr = self.lat.freqs2d.unsqueeze(0).expand(B, *self.lat.freqs2d.shape) / cp[:, 0].view(B, 1, 1)
        c = ctf.compute_ctf(fr, *torch.split(cp[:, 1:], 1, 1)).view(B, -1)[:, self.mask]
        return ind, y, rot, c


def loader(ctx, idx, bs, shuffle, seed=0, workers=4):
    smp = SubsetRandomSampler(idx, generator=torch.Generator().manual_seed(seed)) if shuffle else list(idx)
    return DataLoader(ctx.data, sampler=BatchSampler(smp, batch_size=bs, drop_last=False), batch_size=None,
                      num_workers=workers, multiprocessing_context="spawn" if workers else None,
                      persistent_workers=bool(workers) and shuffle)


@torch.no_grad()
def evaluate(ctx, decs, idx, bs=128):
    """decs: {name: decoder}. Returns per-model metric dict (R0 included as 'R0'). Full fp32 (no TF32)."""
    tf = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    acc = {n: dict(mse=0.0, rel=[], r=[], frc=np.zeros(ctx.nsh), n=0) for n in ["R0", *decs]}
    mse0 = 0.0
    for mb in loader(ctx, idx, bs, False):
        ind, y, rot, c = ctx.batch(mb)
        z = ctx.z[ind]
        s0 = slices(ctx.R0.decoder, ctx.coords, rot, z)
        mse0 += (y ** 2).mean(1).sum().item()
        for n, dec in [("R0", ctx.R0.decoder), *decs.items()]:
            s = s0 if n == "R0" else slices(dec, ctx.coords, rot, z)
            a = acc[n]
            a["mse"] += ((s * c - y) ** 2).mean(1).sum().item()
            a["rel"].append(((s - s0) ** 2).sum(1) / (s0 ** 2).sum(1))
            sc, s0c = s - s.mean(1, keepdim=True), s0 - s0.mean(1, keepdim=True)
            a["r"].append((sc * s0c).sum(1) / (sc.norm(dim=1) * s0c.norm(dim=1)))
            sh = ctx.shell.expand_as(s)
            num = torch.zeros(len(s), ctx.nsh, device=s.device).scatter_add_(1, sh, s * s0)
            p1 = torch.zeros_like(num).scatter_add_(1, sh, s * s)
            p0 = torch.zeros_like(num).scatter_add_(1, sh, s0 * s0)
            a["frc"] += (num / (p1 * p0).sqrt().clamp(min=1e-20)).sum(0).cpu().numpy()
            a["n"] += len(s)
    N = len(idx)
    mse0 /= N
    out = {}
    for n, a in acc.items():
        rel, r = torch.cat(a["rel"]).cpu().numpy(), torch.cat(a["r"]).cpu().numpy()
        out[n] = dict(mse=a["mse"] / N, rel_err_median=float(np.median(rel)), rel_err_mean=float(rel.mean()),
                      r_median=float(np.median(r)), r_mean=float(r.mean()), frc=(a["frc"] / a["n"]).tolist())
    for n in out:
        out[n]["explained_vs_R0"] = (mse0 - out[n]["mse"]) / (mse0 - out["R0"]["mse"])
    out["_mse0"] = mse0
    torch.backends.cuda.matmul.allow_tf32 = tf
    return out


@torch.no_grad()
def examples(ctx, decs, ids):
    """real-space projections (pre-CTF) and data images for a few particles."""
    tf = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    D = ctx.D
    mb = ctx.data[list(ids)]
    ind, y, rot, c = ctx.batch(mb)
    z = ctx.z[ind]

    def real(v):
        img = torch.zeros(len(v), D * D, device=v.device)
        img[:, ctx.mask] = v
        return torch.stack([fft.iht2_center(im.view(D, D)[:-1, :-1]) for im in img]).cpu().numpy()

    out = {"data": real(y), "ctf": c.cpu().numpy()}
    for n, dec in [("R0", ctx.R0.decoder), *decs.items()]:
        s = slices(dec, ctx.coords, rot, z)
        out[n] = real(s)
        out[n + "_ctf"] = real(s * c)
    torch.backends.cuda.matmul.allow_tf32 = tf
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--epoch", type=int, default=50)
    p.add_argument("--particles", required=True)
    p.add_argument("--poses", required=True)
    p.add_argument("--ctf", required=True)
    p.add_argument("--n-heldout", type=int, default=4000)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--eval-at", type=int, nargs="+", default=[1000, 3000, 10000, 30000])
    p.add_argument("--kinds", nargs="+", default=["pe_orig", "raw_k", "pe_new", "pe_s025", "pe_s1", "geom_lowf"])
    p.add_argument("--skip-zero-shot", action="store_true")
    p.add_argument("-o", required=True)
    a = p.parse_args()
    os.makedirs(a.o, exist_ok=True)
    dev = torch.device("cuda")
    ctx = Ctx(a, dev)
    N = ctx.data.N
    rng = np.random.default_rng(0)
    held = np.sort(rng.choice(N, a.n_heldout, replace=False))
    train = np.setdiff1d(np.arange(N), held)
    np.save(f"{a.o}/heldout_idx.npy", held)
    ex_ids = held[:6].tolist()
    ref_freqs = ctx.R0.decoder.rand_freqs.data.detach().cpu().clone()
    res = dict(args=vars(a), zero_shot={}, retrained={})

    # ---- Test 2a: zero-shot PE swaps in the trained R0 decoder ----
    if not a.skip_zero_shot:
        zero_shot(a, ctx, res, held, ex_ids, ref_freqs, dev)
    retrain(a, ctx, res, held, train, ex_ids, ref_freqs, dev)


def zero_shot(a, ctx, res, held, ex_ids, ref_freqs, dev):
    zs = {}
    g = torch.Generator().manual_seed(12345)
    new = torch.randn(ref_freqs.shape, generator=g) * 0.5
    noise = torch.randn(ref_freqs.shape, generator=torch.Generator().manual_seed(777))
    variants = {"new_draw": new, "sigma_x0.5": ref_freqs * 0.5, "sigma_x2": ref_freqs * 2.0,
                "perm_features": ref_freqs[torch.randperm(len(ref_freqs), generator=torch.Generator().manual_seed(5))]}
    for d in (0.01, 0.03, 0.1, 0.3):
        variants[f"perturb_{d}"] = ref_freqs + d * ref_freqs.std() * noise
    for n, f in variants.items():
        dd = make_decoder("pe_orig", ctx.z.shape[1], ctx.D, ref_freqs, 0).to(dev)
        dd.load_state_dict(ctx.R0.decoder.state_dict())
        dd.rand_freqs.data.copy_(f.to(dev))
        dd.eval()
        zs[n] = dd
    t = time.time()
    res["zero_shot"] = evaluate(ctx, zs, held[:2000])
    print("zero-shot done", time.time() - t, flush=True)
    for n in zs:
        print(f"  {n:16s} r={res['zero_shot'][n]['r_median']:.3f} rel={res['zero_shot'][n]['rel_err_median']:.3g} "
              f"E={res['zero_shot'][n]['explained_vs_R0']:.3f}", flush=True)
    np.savez(f"{a.o}/examples_zeroshot.npz", ids=ex_ids, **examples(ctx, zs, ex_ids))
    json.dump(res, open(f"{a.o}/results.json", "w"), indent=1)
    del zs


def retrain(a, ctx, res, held, train, ex_ids, ref_freqs, dev):
    # ---- Test 1 / 2b / control: retrain decoders with z fixed ----
    decs ={k: make_decoder(k, ctx.z.shape[1], ctx.D, ref_freqs, seed=2024).to(dev) for k in a.kinds}
    def groups(d):  # reparam: the learned PE spread log_s gets lr 1e-2 so it is free to grow or collapse
        ls = [p for n, p in d.named_parameters() if n in ("log_s", "log_tau")]
        rest = [p for n, p in d.named_parameters() if n not in ("log_s", "log_tau")]
        return [{"params": rest}] + ([{"params": ls, "lr": 1e-2}] if ls else [])

    opts = {k: torch.optim.Adam(groups(d), lr=1e-4) for k, d in decs.items()}
    curve = {k: [] for k in decs}
    torch.backends.cuda.matmul.allow_tf32 = True  # training steps only (all variants alike); eval is fp32
    step, ep, t0 = 0, 0, time.time()
    while step < a.steps:
        for mb in loader(ctx, train, 64, True, seed=ep):
            ind, y, rot, c = ctx.batch(mb)
            z = ctx.z[ind]
            for k, d in decs.items():
                d.train()
                opts[k].zero_grad()
                loss = F.mse_loss(slices(d, ctx.coords, rot, z) * c, y)
                loss.backward()
                opts[k].step()
                curve[k].append(loss.item())
            step += 1
            if step % 500 == 0:
                print(f"step {step} ({(time.time() - t0) / step:.3f}s/step) " +
                      " ".join(f"{k}={np.mean(v[-500:]):.5f}" for k, v in curve.items()), flush=True)
            if step in a.eval_at or step == a.steps:
                for d in decs.values():
                    d.eval()
                evd = dict(decs)
                if "pe_resample" in decs:
                    evd["pe_resample_ens8"] = EnsembleDec(decs["pe_resample"]).eval()
                ev = evaluate(ctx, evd, held)
                if "pe_reparam" in decs:
                    sr = (decs["pe_reparam"].log_s.exp() / 0.5).detach().cpu().numpy()
                    ev["_reparam_s_over_sigma"] = dict(median=float(np.median(sr)), p90=float(np.percentile(sr, 90)),
                                                       max=float(sr.max()), init=0.05)
                    print(f"  reparam s/sigma median {np.median(sr):.4f} p90 {np.percentile(sr, 90):.4f}", flush=True)
                if "ipe_learn" in decs:
                    tp = decs["ipe_learn"].tau().item() * 128
                    ev["_ipe_learn_tau_px"] = dict(value=tp, init=0.5)
                    print(f"  ipe_learn tau {tp:.4f} px (init 0.5)", flush=True)
                res["retrained"][str(step)] = ev
                print("eval", step, " ".join(f"{k}: r={ev[k]['r_median']:.3f} E={ev[k]['explained_vs_R0']:.3f}"
                                             for k in evd), flush=True)
                json.dump(res, open(f"{a.o}/results.json", "w"), indent=1)
                np.save(f"{a.o}/train_curves.npy", {k: np.array(v) for k, v in curve.items()}, allow_pickle=True)
                np.savez(f"{a.o}/examples_retrained_{step}.npz", ids=ex_ids, **examples(ctx, evd, ex_ids))
            if step >= a.steps:
                break
        ep += 1
    torch.save({k: d.state_dict() for k, d in decs.items()}, f"{a.o}/decoders.pt")
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
