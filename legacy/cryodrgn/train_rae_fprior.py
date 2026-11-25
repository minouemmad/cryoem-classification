#!/usr/bin/env python
"""RAE with a learned factorised prior: each latent coordinate j gets its own 1-D Gaussian mixture p_j, and the
penalty is beta * mean(-sum_j log p_j(z_j)). Widths are floored at sqrt(lambda) (the gradient-penalty scale).
Usage: all train_rae arguments plus the --fprior-* options.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import cryodrgn.commands.train_vae as tv
import train_rae as tr

logger = tv.logger


class FactorPrior(nn.Module):
    def __init__(self, zdim: int, M: int, s_min: float, s_init: float = 0.5, spread: float = 1.5):
        super().__init__()
        self.s_min = s_min
        self.logit = nn.Parameter(torch.zeros(zdim, M))
        self.mean = nn.Parameter(torch.linspace(-spread, spread, M).repeat(zdim, 1) if M > 1 else torch.zeros(zdim, 1))
        raw = math.log(math.expm1(max(s_init - s_min, 1e-3)))
        self.raw_s = nn.Parameter(torch.full((zdim, M), raw))

    def sd(self):
        return self.s_min + F.softplus(self.raw_s)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """z B x zdim -> B x zdim per-coordinate log densities."""
        s = self.sd()
        lp = (-0.5 * ((z[..., None] - self.mean) / s) ** 2 - torch.log(s) - 0.5 * math.log(2 * math.pi)
              + torch.log_softmax(self.logit, -1))
        return torch.logsumexp(lp, -1)

    def summary(self):
        with torch.no_grad():
            w = torch.softmax(self.logit, -1)
            return dict(weights=w.tolist(), means=self.mean.tolist(), sds=self.sd().tolist())


STATE: dict = dict(acc=[])


def loss_function(z_mu, z_logvar, y, ntilts, y_recon, mask, beta, beta_control=None, dose_filters=None):
    args, prior = tr.STATE["args"], STATE["prior"]
    B = y.size(0)
    y = y.view(B, -1)[:, mask]
    if dose_filters is not None:
        y_recon = torch.mul(y_recon, dose_filters[:, mask])
    gen_loss = F.mse_loss(y_recon, y)
    nll = -prior.log_prob(z_mu).sum(1).mean()
    gp = tr.STATE["gp"] if tr.STATE["gp"] is not None else torch.zeros((), device=y.device)
    tr.STATE["gp"] = None
    dc = torch.zeros((), device=y.device)
    if args.decorr > 0 and B > 2:
        zc = z_mu - z_mu.mean(0)
        zc = zc / (zc.std(0) + 1e-6)
        C = zc.T @ zc / (B - 1)
        dc = (C - torch.diag(torch.diag(C))).pow(2).sum() / (C.shape[0] * (C.shape[0] - 1))
    tr.STATE["acc"].append(torch.stack([nll, gp]).detach())
    STATE["acc"].append(dc.detach())
    loss = gen_loss + (args.rae_beta * nll + args.rae_lambda * gp + args.decorr * dc) / mask.sum().float()
    return loss, gen_loss, nll  # cryoDRGN logs the third value as "kld": here -log p(z)


def train_batch(*a, **k):
    out = tr.train_batch(*a, **k)   # main optimiser step (encoder + decoder); prior grads accumulated
    opt = STATE["opt"]
    opt.step()
    opt.zero_grad(set_to_none=True)
    if tr.STATE["step"] % tr.STATE["args"].rae_log_interval == 0 and STATE["acc"]:
        logger.info("# [prior] step %d: decorr=%.4g  per-axis sd of the widest component: %s", tr.STATE["step"],
                    float(torch.stack(STATE["acc"]).mean()),
                    np.round(STATE["prior"].sd().max(1).values.detach().cpu().numpy(), 3).tolist())
        STATE["acc"].clear()
    return out


_save = tv.save_checkpoint


def save_checkpoint(model, optim, epoch, z_mu, z_logvar, out_weights, out_z):
    _save(model, optim, epoch, z_mu, z_logvar, out_weights, out_z)
    p = os.path.join(os.path.dirname(out_weights), os.path.basename(out_weights).replace("weights", "prior").replace(".pkl", ".pt"))
    torch.save(dict(state=STATE["prior"].state_dict(), optim=STATE["opt"].state_dict(),
                    s_min=STATE["prior"].s_min), p)
    with open(p.replace(".pt", ".json"), "w") as f:
        json.dump(STATE["prior"].summary(), f)


class FPriorRAE(tr.HetOnlyRAE):
    def __init__(self, lattice, *a, **k):
        super().__init__(lattice, *a, **k)
        args = tr.STATE["args"]
        prior = FactorPrior(self.zdim, args.prior_components, args.prior_smin).to(lattice.coords.device)
        if args.load_prior:
            prior.load_state_dict(torch.load(args.load_prior, weights_only=False)["state"])
        STATE["prior"] = prior   # kept outside the model: weights.N.pkl stays cryoDRGN-compatible
        STATE["opt"] = torch.optim.Adam(prior.parameters(), lr=args.prior_lr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tv.add_args(parser)
    g = parser.add_argument_group("RAE")
    g.add_argument("--rae-beta", type=float, default=None, help="weight of -log p(z) (default 1/zdim)")
    g.add_argument("--rae-lambda", type=float, required=True)
    g.add_argument("--rae-gp-points", type=int, default=1024)
    g.add_argument("--rae-log-interval", type=int, default=1000)
    g = parser.add_argument_group("factorised prior")
    g.add_argument("--prior-components", type=int, default=4)
    g.add_argument("--prior-smin", type=float, default=None, help="component sd floor (default sqrt(lambda))")
    g.add_argument("--prior-lr", type=float, default=1e-3)
    g.add_argument("--decorr", type=float, default=0.0, help="weight of the mean squared off-diagonal z correlation")
    g.add_argument("--load-prior", default=None)
    args = parser.parse_args()
    if args.rae_beta is None:
        args.rae_beta = 1.0 / args.zdim
    if args.prior_smin is None:
        args.prior_smin = math.sqrt(args.rae_lambda) if args.rae_lambda > 0 else 0.05
    tr.STATE["args"] = args
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "rae.json"), "w") as f:
        json.dump(dict(method="RAE-GP with a learned factorised (per-coordinate 1-D GMM) prior",
                       **{k: v for k, v in vars(args).items()
                          if k.startswith(("rae_", "prior_", "decorr", "load_prior"))}), f, indent=2)
    tv.HetOnlyVAE = FPriorRAE
    tv.run_batch = tr.run_batch
    tv.loss_function = loss_function
    tv.train_batch = train_batch
    tv.save_checkpoint = save_checkpoint
    tv.main(args)


if __name__ == "__main__":
    main()
