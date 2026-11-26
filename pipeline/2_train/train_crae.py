#!/usr/bin/env python
"""Conditional RAE: the cryoDRGN decoder receives [z_free, c_i], where c_i is a fixed per-particle conditioning vector
(the fully conditioned search coordinates S_final of an earlier pipeline run, standardized). The encoder only has to explain
what c does not: z_free is the residual heterogeneity, the input for the next round of the search.

loss = MSE + (beta * 1/2 |z_free|^2 + lambda * ||dD/d[z_free, c]||_F^2) / N_pix          (RAE-GP as in train_rae.py)

The model is stored as an ordinary cryoDRGN HetOnlyVAE with zdim = zfree + d, and z.<epoch>.pkl holds the full [z_free, c]
vectors, so eval_vol / analyze / decode.py work unchanged; the conditioning columns are the last d (cond.json).
Usage: all train_vae arguments (--zdim = number of FREE dimensions) plus --cond FILE.npy (N x d, particle order of the stack)
       --rae-beta / --rae-lambda / --rae-gp-points [--cond-gp-free-only]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np
import torch

import cryodrgn.commands.train_vae as tv
from cryodrgn.pose import PoseTracker

import train_rae as tr  # RAE gradient penalty, loss and logging; patched below to act on the free part only

STATE = tr.STATE
STATE.update(ind=None, cond=None, zfree=None)


class HetOnlyCRAE(tv.HetOnlyVAE):
    def reparameterize(self, mu, logvar):
        return mu

    def encode(self, *img):
        mu, logvar = super().encode(*img)
        zf = STATE["zfree"]
        if self.training:      # training batch: conditioning rows of the particles PoseTracker.get_pose was just asked for
            c = STATE["cond"][STATE["ind"]]
        else:                  # eval_z: filled in afterwards (eval_z below), particle order known there
            c = torch.zeros(mu.size(0), self.zdim - zf, device=mu.device, dtype=mu.dtype)
        return torch.cat([mu[:, :zf], c], 1), logvar


_get_pose = PoseTracker.get_pose


def get_pose(self, ind):
    STATE["ind"] = ind.to(STATE["cond"].device) if torch.is_tensor(ind) else torch.as_tensor(ind, device=STATE["cond"].device)
    return _get_pose(self, ind)


def run_batch(model, lattice, y, rot, ntilts, ctf_params=None, yr=None):
    z_mu, z_logvar, z, y_recon, mask = tr._run_batch(model, lattice, y, rot, ntilts, ctf_params, yr)
    if model.training:
        args = STATE["args"]
        coords = lattice.coords[mask]
        half = len(coords) // 2
        n = min(args.rae_gp_points // 2, half)
        top = coords[torch.randperm(half, device=coords.device)[:n]]
        sub = torch.cat([top, -top.flip(0)])
        out = model(sub / lattice.extent / 2 @ rot, z)
        g = torch.autograd.grad((out * torch.randn_like(out)).sum(), z, create_graph=True)[0]
        if args.cond_gp_free_only:
            g = g[:, : STATE["zfree"]]
        STATE["gp"] = (g ** 2).sum(1).mean() * (half / n)
    return z_mu, z_logvar, z, y_recon, mask


_loss = tr.loss_function


def loss_function(z_mu, *a, **k):
    return _loss(z_mu[:, : STATE["zfree"]], *a, **k)   # 1/2|z|^2 on the free part only


_eval_z = tv.eval_z


def eval_z(*a, **k):
    z_mu, z_logvar = _eval_z(*a, **k)
    c = STATE["cond_eval"]
    assert len(c) == len(z_mu), (len(c), len(z_mu))
    z_mu[:, STATE["zfree"]:] = c
    return z_mu, z_logvar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tv.add_args(parser)
    g = parser.add_argument_group("conditional RAE")
    g.add_argument("--cond", required=True, help="N x d conditioning coordinates (.npy, stack particle order)")
    g.add_argument("--cond-gp-free-only", action="store_true",
                   help="gradient penalty on dD/dz_free only (default: on the full decoder input [z_free, c])")
    g.add_argument("--rae-beta", type=float, default=None, help="weight of 1/2||z_free||^2 (default 1/zfree)")
    g.add_argument("--rae-lambda", type=float, required=True)
    g.add_argument("--rae-gp-points", type=int, default=1024)
    g.add_argument("--rae-log-interval", type=int, default=1000)
    args = parser.parse_args()
    if args.encode_mode == "tilt":
        raise NotImplementedError("tilt series are not supported")
    C = np.load(args.cond).astype(np.float64)
    C = C[:, None] if C.ndim == 1 else C
    mu, sd = C.mean(0), C.std(0)
    C = ((C - mu) / sd).astype(np.float32)
    if args.ind is not None:
        C = C[np.asarray(pickle.load(open(args.ind, "rb")))]
    zf, d = args.zdim, C.shape[1]
    args.zdim = zf + d
    if args.rae_beta is None:
        args.rae_beta = 1.0 / zf
    STATE.update(args=args, zfree=zf, cond_eval=C)
    tr.STATE["args"] = args
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    STATE["cond"] = torch.from_numpy(C).to(dev)
    os.makedirs(args.outdir, exist_ok=True)
    json.dump(dict(method="conditional RAE-GP: decoder input [z_free, c], c fixed per particle (standardized), z.pkl = [z_free, c]",
                   zfree=zf, cond_dim=d, cond=os.path.abspath(args.cond), cond_mean=mu.tolist(), cond_std=sd.tolist(),
                   gp_free_only=args.cond_gp_free_only, **{k: v for k, v in vars(args).items() if k.startswith("rae_")}),
              open(os.path.join(args.outdir, "cond.json"), "w"), indent=2)
    tv.HetOnlyVAE = HetOnlyCRAE
    PoseTracker.get_pose = get_pose
    tv.run_batch = run_batch
    tv.loss_function = loss_function
    tv.train_batch = tr.train_batch
    tv.eval_z = eval_z
    tv.main(args)


if __name__ == "__main__":
    main()
