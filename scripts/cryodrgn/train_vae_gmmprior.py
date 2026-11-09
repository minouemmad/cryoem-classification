#!/usr/bin/env python
"""cryoDRGN train_vae with a learned Gaussian-mixture prior on z: KL(q(z|x) || N(0, I)) -> KL(q(z|x) || p_GMM),
estimated with reparameterised draws; mixture weights, means and variances train jointly with the VAE.
Weights stay loadable by eval_vol / analyze; prior.N.json holds pi, mu, sigma. Usage: train_vae arguments + --gmm-*.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import cryodrgn.commands.train_vae as tv
from cryodrgn import utils
from cryodrgn.models import unparallelize

logger = tv.logger
STATE: dict = dict(prior=None, step=0, acc=[], resume=None)
LOG_2PI = math.log(2 * math.pi)


class GMMPrior(nn.Module):
    def __init__(self, dim, k, init_scale, min_logvar, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.mu = nn.Parameter(torch.randn(k, dim, generator=g) * init_scale)
        self.logvar = nn.Parameter(torch.zeros(k, dim))
        self.logits = nn.Parameter(torch.zeros(k))
        self.min_logvar = min_logvar

    def log_prob(self, z):                                   # z (N, D) -> (N,)
        lv = self.logvar.clamp(min=self.min_logvar)
        d = (z[:, None, :] - self.mu[None]) ** 2 / lv.exp()[None]
        comp = -0.5 * (d.sum(-1) + lv.sum(-1)[None] + z.shape[1] * LOG_2PI)
        return torch.logsumexp(F.log_softmax(self.logits, 0)[None] + comp, dim=1)

    def summary(self):
        pi = F.softmax(self.logits, 0).detach().cpu().numpy()
        mu = self.mu.detach().cpu().numpy(); sd = np.exp(0.5 * self.logvar.clamp(min=self.min_logvar).detach().cpu().numpy())
        dm = np.sqrt(((mu[:, None] - mu[None]) ** 2).sum(-1)); np.fill_diagonal(dm, np.inf)
        return pi, mu, sd, dm


def prior_kl(z_mu, z_logvar):
    prior, S = STATE["prior"], STATE["args"].gmm_samples
    B, D = z_mu.shape
    mu, lv = z_mu.repeat(S, 1), z_logvar.repeat(S, 1)
    z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
    ce = -prior.log_prob(z).view(S, B).mean(0)                 # -E_q log p_GMM
    ent = 0.5 * D * (1 + LOG_2PI) + 0.5 * z_logvar.sum(1)      # H(q)
    gauss = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp(), dim=1)
    STATE["acc"].append(torch.stack([ce.mean(), ent.mean(), gauss.mean()]).detach())
    return (ce - ent).mean()


def loss_function(z_mu, z_logvar, y, ntilts, y_recon, mask, beta, beta_control=None, dose_filters=None):
    """cryoDRGN loss_function with the Gaussian KLD replaced by the mixture-prior KL."""
    B = y.size(0)
    y = y.view(B, -1)[:, mask]
    if dose_filters is not None:
        y_recon = torch.mul(y_recon, dose_filters[:, mask])
    gen_loss = F.mse_loss(y_recon, y)
    kld = prior_kl(z_mu, z_logvar)
    if torch.isnan(kld):
        raise RuntimeError("GMM-prior KL is nan")
    if beta_control is not None:
        raise NotImplementedError("--beta-control is not supported with the GMM prior")
    return gen_loss + beta * kld / mask.sum().float(), gen_loss, kld


_train_batch = tv.train_batch


def train_batch(*a, **k):
    out = _train_batch(*a, **k)
    STATE["step"] += 1
    args = STATE["args"]
    if STATE["step"] % args.gmm_log_interval == 0:
        ce, ent, gauss = torch.stack(STATE["acc"]).mean(0).tolist()
        STATE["acc"].clear()
        pi, mu, sd, dm = STATE["prior"].summary()
        logger.info("# [GMM prior] step %d: KL=%.3f (CE=%.3f, H=%.3f), Gaussian-prior KLD=%.3f, pi=%s, "
                    "mean sd=%.3f, min centre distance=%.3f", STATE["step"], ce - ent, ce, ent, gauss,
                    np.round(np.sort(pi)[::-1], 3).tolist(), float(sd.mean()), float(dm.min()))
    return out


class HetOnlyVAEGMMPrior(tv.HetOnlyVAE):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        args = STATE["args"]
        self.prior = GMMPrior(self.zdim, args.gmm_k, args.gmm_init_scale, args.gmm_min_logvar, args.seed)
        STATE["prior"] = self.prior

    def load_state_dict(self, state_dict, strict=True, **k):
        if not any(key.startswith("prior.") for key in state_dict):
            if STATE["resume"] is None:
                raise ValueError("Checkpoint has no GMM prior; refusing to resume")
            state_dict = dict(state_dict)
            state_dict.update({f"prior.{key}": v for key, v in STATE["resume"].items()})
        return super().load_state_dict(state_dict, strict, **k)


def save_checkpoint(model, optim, epoch, z_mu, z_logvar, out_weights, out_z):
    full = unparallelize(model).state_dict()
    vae = {k: v for k, v in full.items() if not k.startswith("prior.")}
    prior = STATE["prior"].state_dict()
    torch.save(dict(epoch=epoch, model_state_dict=vae, optimizer_state_dict=optim.state_dict(),
                    prior_state_dict=prior), out_weights)
    with open(out_z, "wb") as f:
        pickle.dump(z_mu, f)
        pickle.dump(z_logvar, f)
    args = STATE["args"]
    tag = os.path.basename(out_weights)[len("weights"):-len(".pkl")]
    pi, mu, sd, dm = STATE["prior"].summary()
    json.dump(dict(epoch=epoch, pi=pi.tolist(), mu=mu.tolist(), sd=sd.tolist(), min_centre_distance=float(dm.min())),
              open(os.path.join(args.outdir, f"prior{tag}.json"), "w"), indent=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tv.add_args(parser)
    g = parser.add_argument_group("multimodal Gaussian (mixture) prior")
    g.add_argument("--gmm-k", type=int, default=8, help="mixture components")
    g.add_argument("--gmm-samples", type=int, default=8, help="z draws per particle for the KL estimate")
    g.add_argument("--gmm-init-scale", type=float, default=1.0, help="sd of the initial component means")
    g.add_argument("--gmm-min-logvar", type=float, default=-6.0)
    g.add_argument("--gmm-log-interval", type=int, default=1000)
    args = parser.parse_args()
    STATE["args"] = args
    if args.load == "latest":
        args.load, _ = utils.get_latest_checkpoint(args.outdir)
    if args.load:
        STATE["resume"] = torch.load(args.load, map_location="cpu", weights_only=False)["prior_state_dict"]
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "gmmprior.json"), "w") as f:
        json.dump(dict(method="end-to-end learned diagonal Gaussian-mixture prior, MC KL",
                       **{k: v for k, v in vars(args).items() if k.startswith("gmm_")}), f, indent=2)
    tv.HetOnlyVAE = HetOnlyVAEGMMPrior
    tv.loss_function = loss_function
    tv.train_batch = train_batch
    tv.save_checkpoint = save_checkpoint
    tv.main(args)


if __name__ == "__main__":
    main()
