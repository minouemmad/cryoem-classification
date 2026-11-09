#!/usr/bin/env python
"""cryoDRGN train_vae as a plain deterministic autoencoder: z = E(Y), image MSE only (no KL / prior / penalty).
Checkpoints are zero-padded to the VAE layout, so eval_vol / analyze work unchanged.
Usage: all train_vae arguments (--beta is ignored).
"""
from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F
from torch import nn

import cryodrgn.commands.train_vae as tv
from cryodrgn.models import ResidLinearMLP, MLP

logger = tv.logger
_Base = tv.HetOnlyVAE


class HetOnlyAE(_Base):
    def __init__(self, lattice, qlayers, qdim, players, pdim, in_dim, zdim=1, encode_mode="resid",
                 activation=nn.ReLU, **kw):
        super().__init__(lattice, qlayers, qdim, players, pdim, in_dim, zdim, encode_mode=encode_mode,
                         activation=activation, **kw)
        # draws after the decoder's rand_freqs, so the PE matches the VAE run with the same seed
        if encode_mode == "resid":
            self.encoder = ResidLinearMLP(in_dim, qlayers, qdim, zdim, activation)
        elif encode_mode == "mlp":
            self.encoder = MLP(in_dim, qlayers, qdim, zdim, activation)
        else:
            raise NotImplementedError(encode_mode)

    def encode(self, *img):
        img = (x.view(x.shape[0], -1) for x in img)
        if self.enc_mask is not None:
            img = (x[:, self.enc_mask] for x in img)
        z = self.encoder(*img)
        return z, torch.zeros_like(z)  # second value only keeps cryoDRGN's (mu, logvar) plumbing

    def reparameterize(self, mu, logvar):
        return mu


def loss_function(z_mu, z_logvar, y, ntilts, y_recon, mask, beta, beta_control=None, dose_filters=None):
    B = y.size(0)
    y = y.view(B, -1)[:, mask]
    if dose_filters is not None:
        y_recon = torch.mul(y_recon, dose_filters[:, mask])
    gen_loss = F.mse_loss(y_recon, y)
    # logged in cryoDRGN's "kld" slot for monitoring only (not in the loss): mean ||z||^2
    zz = z_mu.detach().pow(2).sum(1).mean()
    return gen_loss, gen_loss, zz


def _final_linear(enc):
    lin = [m for m in enc.modules() if isinstance(m, (nn.Linear,)) or type(m).__name__ == "MyLinear"]
    return lin[-1]


_save = tv.save_checkpoint


def save_checkpoint(model, optim, epoch, z_mu, z_logvar, out_weights, out_z):
    _save(model, optim, epoch, z_mu, z_logvar, out_weights, out_z)
    ck = torch.load(out_weights, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    m = tv.unparallelize(model)
    names = {id(p): n for n, p in m.named_parameters()}
    last = _final_linear(m.encoder)
    for p in (last.weight, last.bias):
        n = names[id(p)]
        sd[n] = torch.cat([sd[n], torch.zeros_like(sd[n])], 0)
    ck["ae_note"] = "final encoder layer zero-padded zdim -> 2*zdim for stock cryoDRGN loading; mu == z"
    torch.save(ck, out_weights)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tv.add_args(parser)
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "ae.json"), "w") as f:
        json.dump(dict(method="deterministic AE: encoder -> z directly, loss = image MSE only "
                              "(no KL / no z penalty / no gradient penalty); PE = cryoDRGN gaussian, frozen",
                       pe_type=args.pe_type, feat_sigma=args.feat_sigma, seed=args.seed), f, indent=2)
    tv.HetOnlyVAE = HetOnlyAE
    tv.loss_function = loss_function
    tv.save_checkpoint = save_checkpoint
    tv.main(args)


if __name__ == "__main__":
    main()
