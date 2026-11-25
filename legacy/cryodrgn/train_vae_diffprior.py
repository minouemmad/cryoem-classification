#!/usr/bin/env python
"""cryoDRGN train_vae with an end-to-end latent score-based prior (LSGM; Vahdat et al. 2021):
KL(q(z|x) || N(0, I)) -> KL(q(z|x) || p_SGM), with a VP-SDE score model trained jointly (mixed score parameterisation).
Weights stay loadable by eval_vol / analyze; prior.N.pt loads with latent_diffusion.load_model.
Usage: train_vae arguments + --prior-*.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F

import cryodrgn.commands.train_vae as tv
from cryodrgn import utils
from cryodrgn.models import unparallelize

from latent_diffusion import BETA_MAX, BETA_MIN, T_EPS, EpsNet, log_mean_coeff, marginal

logger = tv.logger
STATE: dict = dict(prior=None, ema=None, step=0, acc=[], resume=None)


def log_snr(t: float) -> float:
    lm = log_mean_coeff(t)
    return 2 * lm - math.log(-math.expm1(2 * lm))


LAM_MIN, LAM_MAX = log_snr(1.0), log_snr(T_EPS)
LOG_2PIE = math.log(2 * math.pi * math.e)
SIGMA0_SQ = -math.expm1(2 * log_mean_coeff(T_EPS))


def t_of_lambda(lam):
    """Invert log SNR for the linear VP schedule."""
    lm = -0.5 * F.softplus(-lam)
    d = BETA_MAX - BETA_MIN
    return (-0.5 * BETA_MIN + torch.sqrt(0.25 * BETA_MIN ** 2 - d * lm)) / (0.5 * d)


def prior_kl(z_mu, z_logvar):
    """Per-particle Monte Carlo KL(q || p_SGM) with K (lambda, eps, z0) draws, batch mean."""
    prior, K = STATE["prior"], STATE["args"].prior_samples
    B, D = z_mu.shape
    mu, lv = z_mu.repeat(K, 1), z_logvar.repeat(K, 1)
    z0 = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
    lam = LAM_MIN + (LAM_MAX - LAM_MIN) * torch.rand(B * K, device=mu.device)
    t = t_of_lambda(lam)
    a, s = marginal(t)
    eps = torch.randn_like(z0)
    err = ((prior(a[:, None] * z0 + s[:, None] * eps, t) - eps) ** 2).sum(1)
    ce = 0.5 * (LAM_MAX - LAM_MIN) * err.view(K, B).mean(0) + 0.5 * D * math.log(2 * math.pi * math.e * SIGMA0_SQ)
    ent = 0.5 * D * LOG_2PIE + 0.5 * z_logvar.sum(1)
    gauss = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp(), dim=1)
    STATE["acc"].append(torch.stack([ce.mean(), ent.mean(), gauss.mean()]).detach())
    return (ce - ent).mean()


def loss_function(z_mu, z_logvar, y, ntilts, y_recon, mask, beta, beta_control=None,
                  dose_filters=None):
    """cryoDRGN loss_function with the Gaussian KLD replaced by the SGM-prior KL."""
    B = y.size(0)
    y = y.view(B, -1)[:, mask]
    if dose_filters is not None:
        y_recon = torch.mul(y_recon, dose_filters[:, mask])
    gen_loss = F.mse_loss(y_recon, y)
    kld = prior_kl(z_mu, z_logvar)
    if torch.isnan(kld):
        raise RuntimeError("SGM-prior KL is nan")
    if beta_control is not None:
        raise NotImplementedError("--beta-control is not supported with the SGM prior")
    return gen_loss + beta * kld / mask.sum().float(), gen_loss, kld


_train_batch = tv.train_batch


def train_batch(*a, **k):
    out = _train_batch(*a, **k)
    prior, ema, args = STATE["prior"], STATE["ema"], STATE["args"]
    with torch.no_grad():
        if STATE["step"] == 0:
            ema.to(next(prior.parameters()).device)
        for pe, pm in zip(ema.parameters(), prior.parameters()):
            pe.lerp_(pm, 1 - args.prior_ema)
    STATE["step"] += 1
    if STATE["step"] % args.prior_log_interval == 0:
        ce, ent, gauss = torch.stack(STATE["acc"]).mean(0).tolist()
        STATE["acc"].clear()
        logger.info("# [SGM prior] step %d: KL=%.3f (CE=%.3f, H=%.3f), Gaussian-prior KLD=%.3f, "
                    "mix alpha=%s", STATE["step"], ce - ent, ce, ent, gauss,
                    np.round(torch.sigmoid(prior.mix).tolist(), 3).tolist())
    return out


class HetOnlyVAEDiffPrior(tv.HetOnlyVAE):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        args = STATE["args"]
        self.prior = EpsNet(self.zdim, args.prior_width, args.prior_blocks,
                            mix_init=args.prior_mix_init)
        STATE["prior"] = self.prior
        STATE["ema"] = copy.deepcopy(self.prior).requires_grad_(False)

    def load_state_dict(self, state_dict, strict=True, **k):
        if not any(key.startswith("prior.") for key in state_dict):
            if STATE["resume"] is None:
                raise ValueError("Checkpoint has no SGM prior; refusing to resume")
            state_dict = dict(state_dict)
            state_dict.update({f"prior.{key}": v for key, v in STATE["resume"]["prior"].items()})
            STATE["ema"].load_state_dict(STATE["resume"]["ema"])
        return super().load_state_dict(state_dict, strict, **k)


def save_checkpoint(model, optim, epoch, z_mu, z_logvar, out_weights, out_z):
    full = unparallelize(model).state_dict()
    vae = {k: v for k, v in full.items() if not k.startswith("prior.")}
    prior = STATE["prior"].state_dict()
    ema = STATE["ema"].state_dict()
    torch.save(dict(epoch=epoch, model_state_dict=vae, optimizer_state_dict=optim.state_dict(),
                    prior_state_dict=prior, prior_ema_state_dict=ema), out_weights)
    with open(out_z, "wb") as f:
        pickle.dump(z_mu, f)
        pickle.dump(z_logvar, f)
    args = STATE["args"]
    tag = os.path.basename(out_weights)[len("weights"):-len(".pkl")]  # ".N" or ""
    torch.save(dict(ema=ema, model=prior, dim=z_mu.shape[1], width=args.prior_width,
                    blocks=args.prior_blocks, mix=True, mean=np.zeros(z_mu.shape[1]), scale=1.0,
                    run=os.path.abspath(args.outdir), epoch=epoch),
               os.path.join(args.outdir, f"prior{tag}.pt"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    tv.add_args(parser)
    g = parser.add_argument_group("SGM prior (Path B)")
    g.add_argument("--prior-width", type=int, default=512)
    g.add_argument("--prior-blocks", type=int, default=4)
    g.add_argument("--prior-samples", type=int, default=8,
                   help="(lambda, eps, z) draws per particle for the KL estimate")
    g.add_argument("--prior-mix-init", type=float, default=-3.0,
                   help="logit of the initial mixed-score weight alpha")
    g.add_argument("--prior-ema", type=float, default=0.9999)
    g.add_argument("--prior-log-interval", type=int, default=1000)
    args = parser.parse_args()
    STATE["args"] = args
    if args.load == "latest":
        args.load, _ = utils.get_latest_checkpoint(args.outdir)
    if args.load:
        ck = torch.load(args.load, map_location="cpu", weights_only=False)
        STATE["resume"] = dict(prior=ck["prior_state_dict"], ema=ck["prior_ema_state_dict"])
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "diffprior.json"), "w") as f:
        json.dump(dict(method="LSGM end-to-end SGM prior", sde="VP", beta_min=BETA_MIN,
                       beta_max=BETA_MAX, t_eps=T_EPS, lambda_range=[LAM_MIN, LAM_MAX],
                       **{k: v for k, v in vars(args).items() if k.startswith("prior_")}),
                  f, indent=2)
    tv.HetOnlyVAE = HetOnlyVAEDiffPrior
    tv.loss_function = loss_function
    tv.train_batch = train_batch
    tv.save_checkpoint = save_checkpoint
    tv.main(args)


if __name__ == "__main__":
    main()
