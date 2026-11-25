#!/usr/bin/env python
"""cryoDRGN train_vae as a regularized deterministic autoencoder (RAE-GP; Ghosh et al., ICLR 2020):
loss = MSE + (beta * 1/2 |z|^2 + lambda * ||dD/dz||_F^2) / N_pix, with z = mu(x).
Checkpoints load unchanged in eval_vol / analyze. Usage: all train_vae arguments plus --rae-beta / --rae-lambda / --rae-gp-points.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F

import cryodrgn.commands.train_vae as tv

logger = tv.logger
STATE: dict = dict(step=0, acc=[], gp=None)


class HetOnlyRAE(tv.HetOnlyVAE):
    def reparameterize(self, mu, logvar):
        return mu


_run_batch = tv.run_batch


def run_batch(model, lattice, y, rot, ntilts, ctf_params=None, yr=None):
    z_mu, z_logvar, z, y_recon, mask = _run_batch(model, lattice, y, rot, ntilts, ctf_params, yr)
    if model.training:
        args = STATE["args"]
        coords = lattice.coords[mask]
        # The Fourier decoder evaluates the top half and mirrors it (Friedel symmetry), so
        # sample Friedel pairs: top-half points followed by their reversed negatives.
        half = len(coords) // 2
        n = min(args.rae_gp_points // 2, half)
        top = coords[torch.randperm(half, device=coords.device)[:n]]
        sub = torch.cat([top, -top.flip(0)])
        out = model(sub / lattice.extent / 2 @ rot, z)
        v = torch.randn_like(out)
        g = torch.autograd.grad((out * v).sum(), z, create_graph=True)[0]
        STATE["gp"] = (g ** 2).sum(1).mean() * (half / n)
    return z_mu, z_logvar, z, y_recon, mask


def loss_function(z_mu, z_logvar, y, ntilts, y_recon, mask, beta, beta_control=None,
                  dose_filters=None):
    if beta_control is not None:
        raise NotImplementedError("--beta-control is not supported for the RAE")
    args = STATE["args"]
    B = y.size(0)
    y = y.view(B, -1)[:, mask]
    if dose_filters is not None:
        y_recon = torch.mul(y_recon, dose_filters[:, mask])
    gen_loss = F.mse_loss(y_recon, y)
    lz = 0.5 * z_mu.pow(2).sum(1).mean()
    gp = STATE["gp"] if STATE["gp"] is not None else torch.zeros((), device=y.device)
    STATE["gp"] = None
    STATE["acc"].append(torch.stack([lz, gp]).detach())
    loss = gen_loss + (args.rae_beta * lz + args.rae_lambda * gp) / mask.sum().float()
    return loss, gen_loss, lz  # cryoDRGN logs the third value as "kld": here 1/2||z||^2


_train_batch = tv.train_batch


def train_batch(*a, **k):
    out = _train_batch(*a, **k)
    STATE["step"] += 1
    if STATE["step"] % STATE["args"].rae_log_interval == 0:
        lz, gp = torch.stack(STATE["acc"]).mean(0).tolist()
        STATE["acc"].clear()
        args = STATE["args"]
        logger.info("# [RAE] step %d: 1/2|z|^2=%.3f, GP=||J||_F^2=%.4g (beta*Lz=%.4g, lambda*GP=%.4g)",
                    STATE["step"], lz, gp, args.rae_beta * lz, args.rae_lambda * gp)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    tv.add_args(parser)
    g = parser.add_argument_group("RAE")
    g.add_argument("--rae-beta", type=float, default=None,
                   help="weight of 1/2||z||^2 (default: cryoDRGN beta, 1/zdim)")
    g.add_argument("--rae-lambda", type=float, required=True,
                   help="gradient-penalty weight (sigma^2 of the replaced VAE noise)")
    g.add_argument("--rae-gp-points", type=int, default=1024)
    g.add_argument("--rae-log-interval", type=int, default=1000)
    args = parser.parse_args()
    if args.rae_beta is None:
        args.rae_beta = 1.0 / args.zdim
    STATE["args"] = args
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "rae.json"), "w") as f:
        json.dump(dict(method="RAE-GP (Ghosh et al. 2020), deterministic encoder z = mu(x)",
                       **{k: v for k, v in vars(args).items() if k.startswith("rae_")}), f, indent=2)
    tv.HetOnlyVAE = HetOnlyRAE
    tv.run_batch = run_batch
    tv.loss_function = loss_function
    tv.train_batch = train_batch
    tv.main(args)


if __name__ == "__main__":
    main()
