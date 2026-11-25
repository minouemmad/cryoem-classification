#!/usr/bin/env python
"""Score every decoder of a pe_tests.py output directory against the held-out particle images (MSE, explained
variance, per-particle correlation). Writes <dir>/data_metrics.json.
Usage: pe_eval_data.py <pe_tests_dir> [--zero-shot]
"""
import argparse
import json
import time
from types import SimpleNamespace

import numpy as np
import torch

import pe_tests as T


@torch.no_grad()
def evaluate_data(ctx, decs, idx, bs=128):
    torch.backends.cuda.matmul.allow_tf32 = False
    names = ["R0", *decs]
    acc = {n: dict(mse=0.0, r=[], num=torch.zeros(ctx.nsh, dtype=torch.float64, device=ctx.dev),
                   pp=torch.zeros(ctx.nsh, dtype=torch.float64, device=ctx.dev)) for n in names}
    py = torch.zeros(ctx.nsh, dtype=torch.float64, device=ctx.dev)
    mse0 = 0.0
    for mb in T.loader(ctx, idx, bs, False):
        ind, y, rot, c = ctx.batch(mb)
        z = ctx.z[ind]
        sh = ctx.shell.expand_as(y)
        mse0 += (y ** 2).mean(1).sum().item()
        py += torch.zeros(len(y), ctx.nsh, device=y.device).scatter_add_(1, sh, y * y).sum(0).double()
        yc = y - y.mean(1, keepdim=True)
        for n, dec in [("R0", ctx.R0.decoder), *decs.items()]:
            p = T.slices(dec, ctx.coords, rot, z) * c
            a = acc[n]
            a["mse"] += ((p - y) ** 2).mean(1).sum().item()
            pc = p - p.mean(1, keepdim=True)
            a["r"].append((pc * yc).sum(1) / (pc.norm(dim=1) * yc.norm(dim=1)).clamp(min=1e-20))
            a["num"] += torch.zeros(len(y), ctx.nsh, device=y.device).scatter_add_(1, sh, p * y).sum(0).double()
            a["pp"] += torch.zeros(len(y), ctx.nsh, device=y.device).scatter_add_(1, sh, p * p).sum(0).double()
    N = len(idx)
    mse0 /= N
    out = {}
    for n, a in acc.items():
        r = torch.cat(a["r"]).cpu().numpy()
        frc = (a["num"] / (a["pp"] * py).sqrt().clamp(min=1e-30)).cpu().numpy()
        out[n] = dict(mse=a["mse"] / N, expl=1 - a["mse"] / N / mse0, r_median=float(np.median(r)),
                      r_mean=float(r.mean()), frc=frc.tolist())
    out["_mse0"] = mse0
    out["_n"] = N
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("--zero-shot", action="store_true", help="also rebuild and score the zero-shot PE swaps")
    a = p.parse_args()
    R = json.load(open(f"{a.dir}/results.json"))
    args = SimpleNamespace(**R["args"])
    dev = torch.device("cuda")
    ctx = T.Ctx(args, dev)
    held = np.load(f"{a.dir}/heldout_idx.npy")
    ref = ctx.R0.decoder.rand_freqs.data.detach().cpu().clone()
    zd = ctx.z.shape[1]
    decs = {}
    if a.zero_shot:
        new = torch.randn(ref.shape, generator=torch.Generator().manual_seed(12345)) * 0.5
        noise = torch.randn(ref.shape, generator=torch.Generator().manual_seed(777))
        var = {"zs_new_draw": new, "zs_sigma_x0.5": ref * 0.5, "zs_sigma_x2": ref * 2.0}
        for d in (0.01, 0.03, 0.1, 0.3):
            var[f"zs_perturb_{d}"] = ref + d * ref.std() * noise
        for n, f in var.items():
            dd = T.make_decoder("pe_orig", zd, ctx.D, ref, 0).to(dev)
            dd.load_state_dict(ctx.R0.decoder.state_dict())
            dd.rand_freqs.data.copy_(f.to(dev))
            decs[n] = dd.eval()
    sd = torch.load(f"{a.dir}/decoders.pt", map_location=dev)
    for k, s in sd.items():
        d = T.make_decoder(k, zd, ctx.D, ref, seed=2024).to(dev)
        d.load_state_dict(s)
        decs[k] = d.eval()
        if k == "pe_resample":
            decs["pe_resample_ens8"] = T.EnsembleDec(d).eval()
    t = time.time()
    out = evaluate_data(ctx, decs, held)
    out["_steps"] = max(int(s) for s in R["retrained"])
    # learning curves vs the images, from the mse logged at each evaluation (same held-out set, same mse0)
    out["_curve"] = {k: {s: 1 - R["retrained"][s][k]["mse"] / R["retrained"][s]["_mse0"] for s in R["retrained"]}
                     for k in R["retrained"][str(out["_steps"])] if k[0] != "_"}
    json.dump(out, open(f"{a.dir}/data_metrics.json", "w"), indent=1)
    print(f"done in {time.time() - t:.0f}s  blank mse {out['_mse0']:.5f}")
    for n, v in out.items():
        if n[0] != "_":
            f = v["frc"]
            print(f"  {n:18s} expl={v['expl'] * 100:6.3f} %  r={v['r_median']:.4f}  FRC@20/10/5A="
                  f"{f[13]:.3f}/{f[27]:.3f}/{f[53]:.3f}")


if __name__ == "__main__":
    main()
