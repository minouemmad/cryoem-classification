#!/usr/bin/env python
"""RAE with a deformation decoder: z moves a fixed template of 3-D Gaussians (rigid-domain skinning) instead of
generating free density, so mass is conserved. Images are analytic Fourier slices of the moved Gaussians; poses, CTF,
loss and RAE penalties are those of train_rae.py.
Subcommands: init (fit the template to a decoded map), train, eval (decode volumes).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from cryodrgn import fft
from cryodrgn.lattice import Lattice
from cryodrgn.models import Decoder

AMP_SCALE = 10.0   # parameter scalings = per-parameter learning-rate multipliers under Adam
DOM_SCALE = 100.0
ROT_SCALE = 0.1    # rad per unit network output
T_SCALE = 2.0      # px per unit network output


def rotvec_to_matrix(v: torch.Tensor) -> torch.Tensor:
    """Axis-angle (...,3) -> rotation matrices (...,3,3), Rodrigues, stable near 0."""
    th2 = (v * v).sum(-1, keepdim=True)
    th = torch.sqrt(th2 + 1e-12)
    a = torch.where(th2 < 1e-8, 1 - th2 / 6, torch.sin(th) / th)
    b = torch.where(th2 < 1e-8, 0.5 - th2 / 24, (1 - torch.cos(th)) / (th2 + 1e-12))
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    Kx = torch.stack([o, -z, y, z, o, -x, -y, x, o], -1).view(*v.shape[:-1], 3, 3)
    eye = torch.eye(3, device=v.device, dtype=v.dtype).expand_as(Kx)
    return eye + a[..., None] * Kx + b[..., None] * (Kx @ Kx)


def _slice_chunk(x, pos, amp):
    """x B x P x 3 (cycles/px), pos B x n x 3 (px), amp n -> (sum a cos, sum a sin), B x P."""
    th = (2 * math.pi) * torch.bmm(x, pos.transpose(1, 2))
    return torch.cos(th) @ amp, torch.sin(th) @ amp


class GaussDeformDecoder(Decoder):
    def __init__(self, zdim: int, D: int, template: dict, n_domains: int, hidden: int = 256,
                 layers: int = 3, chunk: int = 512, sign: float = 1.0):
        super().__init__()
        self.zdim, self.D, self.K, self.chunk, self.sign = zdim, D, n_domains, chunk, sign
        r0 = torch.as_tensor(template["centres"], dtype=torch.float32)
        self.r0 = nn.Parameter(r0)
        self.loga = nn.Parameter(torch.as_tensor(template["log_amp"], dtype=torch.float32) / AMP_SCALE)
        self.logs = nn.Parameter(torch.tensor(float(template["log_sigma"])))
        self.dom = nn.Parameter(torch.as_tensor(template["domain_logits"], dtype=torch.float32) / DOM_SCALE)
        self.register_buffer("knn", torch.as_tensor(template["knn"], dtype=torch.long))
        mods, d = [], zdim
        for _ in range(layers - 1):
            mods += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        last = nn.Linear(d, 6 * n_domains)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.net = nn.Sequential(*mods, last)

    # -- geometry
    def weights(self) -> torch.Tensor:
        return torch.softmax(DOM_SCALE * self.dom, -1)

    def motions(self, z):
        out = self.net(z).view(z.shape[0], self.K, 6)
        return rotvec_to_matrix(ROT_SCALE * out[..., :3]), T_SCALE * out[..., 3:]

    def positions(self, z: torch.Tensor) -> torch.Tensor:
        w = self.weights()
        c = (w.T @ self.r0) / w.sum(0)[:, None]                       # K x 3 pivots
        R, t = self.motions(z)                                        # B K 3 3, B K 3
        rel = self.r0[None] - c[:, None]                              # K N 3
        moved = torch.einsum("bkij,knj->bkni", R, rel) + (c[None] + t)[:, :, None]
        return torch.einsum("nk,bkni->bni", w, moved)

    def smoothness(self) -> torch.Tensor:
        w = self.weights()
        return ((w[:, None, :] - w[self.knn]) ** 2).sum(-1).mean()

    # -- Fourier slice
    def ft(self, x: torch.Tensor, pos: torch.Tensor):
        """x B x P x 3, pos B x N x 3 -> Re, Im of the template FT at x (B x P)."""
        amp = torch.exp(AMP_SCALE * self.loga)
        C = torch.zeros(x.shape[:2], device=x.device)
        S = torch.zeros_like(C)
        for i in range(0, pos.shape[1], self.chunk):
            sl = slice(i, i + self.chunk)
            if torch.is_grad_enabled():
                c, s = checkpoint(_slice_chunk, x, pos[:, sl], amp[sl], use_reentrant=False)
            else:
                c, s = _slice_chunk(x, pos[:, sl], amp[sl])
            C, S = C + c, S + s
        env = torch.exp(-2 * math.pi ** 2 * torch.exp(2 * self.logs) * (x * x).sum(-1))
        return env * C, -self.sign * env * S

    def decode(self, lattice: torch.Tensor) -> torch.Tensor:
        squeeze = lattice.dim() == 2
        if squeeze:
            lattice = lattice[None]
        x, z = lattice[..., :3], lattice[:, 0, 3:]
        pos = self.positions(z) if self.zdim else self.r0[None].expand(len(x), -1, -1)
        re, im = self.ft(x, pos)
        out = torch.stack([re, im], -1)
        return out[0] if squeeze else out

    def forward(self, lattice: torch.Tensor) -> torch.Tensor:
        # identical to cryoDRGN FTPositionalDecoder.forward: evaluate the top half, mirror the rest
        c = lattice.shape[-2] // 2
        cc = c + 1 if lattice.shape[-2] % 2 == 1 else c
        image = torch.empty(lattice.shape[:-1], device=lattice.device)
        top = self.decode(lattice[..., 0:cc, :])
        image[..., 0:cc] = top[..., 0] - top[..., 1]
        image[..., cc:] = (top[..., 0] + top[..., 1])[..., np.arange(c - 1, -1, -1)]
        return image

    def eval_volume(self, coords, D, extent, norm, zval=None):
        vol_f = torch.zeros((D, D, D), dtype=torch.float32, device=coords.device)
        z = None if zval is None else torch.tensor(zval, dtype=torch.float32, device=coords.device)
        with torch.no_grad():
            pos = self.positions(z[None]) if z is not None else self.r0[None]
            for i, dz in enumerate(np.linspace(-extent, extent, D, endpoint=True, dtype=np.float32)):
                x = coords + torch.tensor([0, 0, dz], device=coords.device)
                keep = x.pow(2).sum(dim=1) <= extent ** 2
                re, im = self.ft(x[keep][None], pos)
                sl = torch.zeros(D ** 2, device=coords.device)
                sl[keep] = (re - im)[0]
                vol_f[i] = sl.view(D, D)
        vol_f = vol_f * norm[1] + norm[0]
        return fft.ihtn_center(vol_f[:-1, :-1, :-1])


# ---------------------------------------------------------------------------- init
def cmd_init(a):
    import mrcfile
    import yaml
    from sklearn.cluster import KMeans, MiniBatchKMeans
    from sklearn.neighbors import NearestNeighbors

    dev = torch.device("cuda")
    cfg = yaml.safe_load(open(a.config))
    norm = [float(v) for v in cfg["dataset_args"]["norm"]]
    Dl = int(cfg["lattice_args"]["D"])
    lat = Lattice(Dl, extent=0.5, device=dev)
    M = mrcfile.read(a.map).astype(np.float32)
    n = M.shape[0]
    assert n == Dl - 1
    # target in training units: eval_volume inverts y -> ihtn_center(y * norm1 + norm0)
    Hm = fft.htn_center(torch.tensor(M, device=dev))
    rt = fft.ihtn_center(Hm)
    print("ht round trip max err", float((rt - torch.tensor(M, device=dev)).abs().max()))
    T = torch.zeros((Dl, Dl, Dl), device=dev)
    T[:-1, :-1, :-1] = (Hm - norm[0]) / norm[1]
    dzs = torch.tensor(np.linspace(-0.5, 0.5, Dl, endpoint=True, dtype=np.float32), device=dev)
    allx = (lat.coords[None] + torch.stack([torch.zeros_like(dzs), torch.zeros_like(dzs), dzs], 1)[:, None]).view(-1, 3)
    ally = T.view(-1)
    inball = allx.pow(2).sum(1) <= 0.25
    allx, ally = allx[inball], ally[inball]

    # convention probe: a single Gaussian at known (x,y,z) px through eval_volume
    probe = dict(centres=np.array([[10.0, 20.0, 30.0]]), log_amp=np.zeros(1), log_sigma=0.0,
                 domain_logits=np.zeros((1, 1)), knn=np.zeros((1, 1)))
    for sign in (1.0, -1.0):
        dec = GaussDeformDecoder(0, Dl, probe, 1, sign=sign).to(dev).eval()
        v = dec.eval_volume(lat.coords, Dl, 0.5, [0.0, 1.0]).cpu().numpy()
        idx = np.unravel_index(np.argmax(v), v.shape)
        print(f"probe sign {sign:+.0f}: peak at array index {idx} (centre {n // 2})")
    # mapping found: array[iz, iy, ix] <-> r = (ix, iy, iz) - n/2 for the sign whose peak is at
    # (n/2+30, n/2+20, n/2+10); chosen below, asserted
    sign = None
    for s in (1.0, -1.0):
        dec = GaussDeformDecoder(0, Dl, probe, 1, sign=s).to(dev).eval()
        v = dec.eval_volume(lat.coords, Dl, 0.5, [0.0, 1.0]).cpu().numpy()
        if np.unravel_index(np.argmax(v), v.shape) == (n // 2 + 30, n // 2 + 20, n // 2 + 10):
            sign = s
    assert sign is not None, "no sign reproduces the expected (z,y,x) peak; check conventions"
    print("using sign", sign)

    # centres: density-weighted k-means of voxels above threshold
    thr = a.thr_sd * M.std()
    iz, iy, ix = np.nonzero(M > thr)
    pts = np.stack([ix, iy, iz], 1).astype(np.float32) - n // 2
    wts = M[iz, iy, ix]
    print(f"{len(pts)} voxels above {a.thr_sd} sd; fitting {a.n} centres")
    km = MiniBatchKMeans(a.n, random_state=0, batch_size=8192, n_init=3, max_iter=300).fit(pts, sample_weight=wts)
    C = km.cluster_centers_.astype(np.float32)
    tmpl = dict(centres=C, log_amp=np.zeros(a.n), log_sigma=math.log(a.sigma),
                domain_logits=np.zeros((a.n, 1)), knn=np.zeros((a.n, 1)))
    dec = GaussDeformDecoder(0, Dl, tmpl, 1, sign=sign).to(dev)
    # optimise centres, amplitudes, width against the template's Fourier coefficients
    opt = torch.optim.Adam([dict(params=[dec.r0], lr=0.02), dict(params=[dec.loga, dec.logs], lr=0.01)])
    # sensible starting amplitude: least squares scale
    sel = torch.randperm(len(allx), device=dev)[:20000]
    with torch.no_grad():
        re, im = dec.ft(allx[sel][None], dec.r0[None])
        pred = (re - im)[0]
        scale = float((pred * ally[sel]).sum() / (pred * pred).sum())
        dec.loga.fill_(math.log(max(scale, 1e-12)) / AMP_SCALE)
    for it in range(a.iters):
        sel = torch.randperm(len(allx), device=dev)[:a.points]
        x = allx[sel]
        re, im = dec.ft(x[None], dec.r0[None])
        loss = ((re - im)[0] - ally[sel]).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 200 == 0 or it == a.iters - 1:
            print(f"it {it} loss {loss.item():.4g} sigma {dec.logs.exp().item():.3f} px", flush=True)
    dec.eval()
    V = dec.eval_volume(lat.coords, Dl, 0.5, norm).cpu().numpy()
    r = np.corrcoef(V.ravel(), M.ravel())[0, 1]
    rm = np.corrcoef(V[::-1].ravel(), M.ravel())[0, 1]
    print(f"template vs map r = {r:.4f} (z-mirrored {rm:.4f})")
    C = dec.r0.detach().cpu().numpy()
    km = KMeans(a.domains, random_state=0, n_init=10).fit(C)
    logits = np.full((a.n, a.domains), -2.0, np.float32)
    logits[np.arange(a.n), km.labels_] = 2.0
    knn = NearestNeighbors(n_neighbors=a.knn + 1).fit(C).kneighbors(C, return_distance=False)[:, 1:]
    out = dict(centres=C, log_amp=(AMP_SCALE * dec.loga).detach().cpu().numpy(),
               log_sigma=float(dec.logs), domain_logits=logits, knn=knn, sign=sign,
               map=os.path.abspath(a.map), fit_r=float(r), fit_r_mirrored=float(rm), norm=norm, D=Dl)
    torch.save(out, a.out)
    with mrcfile.new(a.out.replace(".pt", "_fit.mrc"), overwrite=True) as f:
        f.set_data(V.astype(np.float32))
        f.voxel_size = a.apix
    print("saved", a.out)


# ---------------------------------------------------------------------------- train
def cmd_train(argv):
    sys.argv = [sys.argv[0]] + argv
    import cryodrgn.commands.train_vae as tv
    import train_rae as tr

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tv.add_args(parser)
    g = parser.add_argument_group("RAE")
    g.add_argument("--rae-beta", type=float, default=None)
    g.add_argument("--rae-lambda", type=float, required=True)
    g.add_argument("--rae-gp-points", type=int, default=1024)
    g.add_argument("--rae-log-interval", type=int, default=1000)
    g = parser.add_argument_group("deformation decoder")
    g.add_argument("--template", required=True, help="template.pt from `init`")
    g.add_argument("--domains", type=int, default=None, help="default: as in the template")
    g.add_argument("--smooth", type=float, default=1.0, help="weight of the kNN domain-smoothness penalty")
    g.add_argument("--motion-hidden", type=int, default=256)
    g.add_argument("--chunk", type=int, default=512)
    args = parser.parse_args()
    if args.rae_beta is None:
        args.rae_beta = 1.0 / args.zdim
    tr.STATE["args"] = args
    tmpl = torch.load(args.template, weights_only=False)
    if args.domains is not None and args.domains != tmpl["domain_logits"].shape[1]:
        raise ValueError("re-run init with --domains to change K")
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "deform.json"), "w") as f:
        json.dump(dict(method="RAE-GP with Gaussian-template rigid-domain deformation decoder",
                       template=os.path.abspath(args.template), n_gauss=int(len(tmpl["centres"])),
                       domains=int(tmpl["domain_logits"].shape[1]), smooth=args.smooth,
                       motion_hidden=args.motion_hidden,
                       **{k: v for k, v in vars(args).items() if k.startswith("rae_")}), f, indent=2)

    class DeformRAE(tr.HetOnlyRAE):
        def __init__(self, lattice, *a, **k):
            super().__init__(lattice, *a, **k)
            self.decoder = GaussDeformDecoder(self.zdim, lattice.D, tmpl, tmpl["domain_logits"].shape[1],
                                              hidden=args.motion_hidden, chunk=args.chunk, sign=tmpl["sign"])

    _loss = tr.loss_function

    def loss_function(*a, **k):
        loss, gen_loss, lz = _loss(*a, **k)
        mask = a[5] if len(a) > 5 else k["mask"]
        dec = tr.STATE["model"].decoder
        return loss + args.smooth * dec.smoothness() / mask.sum().float(), gen_loss, lz

    _run = tr.run_batch

    def run_batch(model, *a, **k):
        tr.STATE["model"] = tv.unparallelize(model)
        return _run(model, *a, **k)

    tv.HetOnlyVAE = DeformRAE
    tv.run_batch = run_batch
    tv.loss_function = loss_function
    tv.train_batch = tr.train_batch
    tv.main(args)


# ---------------------------------------------------------------------------- decode
def load_model(run, epoch, device):
    import yaml
    cfg = yaml.safe_load(open(os.path.join(run, "config.yaml")))
    meta = json.load(open(os.path.join(run, "deform.json")))
    tmpl = torch.load(meta["template"], weights_only=False)
    Dl = int(cfg["lattice_args"]["D"])
    zdim = int(cfg["model_args"]["zdim"])
    dec = GaussDeformDecoder(zdim, Dl, tmpl, meta["domains"], hidden=meta["motion_hidden"], sign=tmpl["sign"])
    sd = torch.load(os.path.join(run, f"weights.{epoch}.pkl"), map_location="cpu", weights_only=False)["model_state_dict"]
    dec.load_state_dict({k[len("decoder."):]: v for k, v in sd.items() if k.startswith("decoder.")})
    norm = [float(v) for v in cfg["dataset_args"]["norm"]]
    return dec.to(device).eval(), Lattice(Dl, extent=0.5, device=device), norm


def cmd_decode(a):
    import mrcfile
    dev = torch.device(f"cuda:{a.device}")
    dec, lat, norm = load_model(a.run, a.epoch, dev)
    Z = np.loadtxt(a.zfile, ndmin=2)
    os.makedirs(a.outdir, exist_ok=True)
    for i, z in enumerate(Z):
        V = dec.eval_volume(lat.coords, lat.D, 0.5, norm, z).cpu().numpy().astype(np.float32)
        with mrcfile.new(os.path.join(a.outdir, f"vol_{i:03d}.mrc"), overwrite=True) as f:
            f.set_data(V)
            f.voxel_size = a.apix
        if a.positions:
            with torch.no_grad():
                np.save(os.path.join(a.outdir, f"pos_{i:03d}.npy"),
                        dec.positions(torch.tensor(z[None], dtype=torch.float32, device=dev))[0].cpu().numpy())
    with torch.no_grad():
        np.save(os.path.join(a.outdir, "domain_weights.npy"), dec.weights().cpu().numpy())
        np.save(os.path.join(a.outdir, "template_centres.npy"), dec.r0.cpu().numpy())
    print(f"decoded {len(Z)} volumes -> {a.outdir}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        return cmd_train(sys.argv[2:])
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("init")
    q.add_argument("--map", required=True, help="cryoDRGN-decoded map (native box, cryoDRGN frame)")
    q.add_argument("--config", required=True, help="config.yaml of the run whose norm/lattice to use")
    q.add_argument("-n", type=int, default=4000, help="number of Gaussians")
    q.add_argument("--sigma", type=float, default=1.2, help="initial width (px)")
    q.add_argument("--thr-sd", type=float, default=2.0)
    q.add_argument("--domains", type=int, default=8)
    q.add_argument("--knn", type=int, default=8)
    q.add_argument("--iters", type=int, default=3000)
    q.add_argument("--points", type=int, default=20000)
    q.add_argument("--apix", type=float, default=2.075)
    q.add_argument("-o", "--out", required=True)
    q.set_defaults(func=cmd_init)
    q = sub.add_parser("decode")
    q.add_argument("--run", required=True)
    q.add_argument("--epoch", type=int, required=True)
    q.add_argument("--zfile", required=True)
    q.add_argument("--apix", type=float, default=2.075)
    q.add_argument("--device", type=int, default=0)
    q.add_argument("--positions", action="store_true")
    q.add_argument("-o", "--outdir", required=True)
    q.set_defaults(func=cmd_decode)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
