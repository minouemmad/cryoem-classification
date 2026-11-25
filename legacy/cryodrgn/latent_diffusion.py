#!/usr/bin/env python
"""Post-hoc VP-SDE score model on frozen cryoDRGN latents, with mode discovery.
Subcommands: train (eps-prediction MLP on z.N.pkl), analyze (fidelity, samples, scale-space mode search),
decode (cryodrgn eval_vol on the discovered modes).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

LOG = logging.getLogger("latent_diffusion")
BETA_MIN, BETA_MAX, T_EPS = 0.1, 20.0, 1e-5


# ----------------------------------------------------------------------------- SDE
def log_mean_coeff(t):
    return -0.25 * t ** 2 * (BETA_MAX - BETA_MIN) - 0.5 * t * BETA_MIN


def marginal(t):
    """alpha(t), sigma(t) of the VP SDE perturbation kernel."""
    lm = log_mean_coeff(t)
    return torch.exp(lm), torch.sqrt(-torch.expm1(2.0 * lm))


def beta(t):
    return BETA_MIN + t * (BETA_MAX - BETA_MIN)


# --------------------------------------------------------------------------- model
class TimeEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half = dim // 2
        self.register_buffer("freqs", torch.exp(-math.log(10000.0) * torch.arange(half) / half))

    def forward(self, t):
        x = 999.0 * t[:, None] * self.freqs[None]
        return torch.cat([x.sin(), x.cos()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, width, temb):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.t = nn.Linear(temb, width)
        self.fc2 = nn.Linear(width, width)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, h, te):
        x = nn.functional.silu(self.fc1(self.norm(h)) + self.t(te))
        return h + self.fc2(nn.functional.silu(x))


class EpsNet(nn.Module):
    """eps_theta(x, t); score = -eps / sigma(t). MLP, no spatial structure.

    mix_init (logit) enables the LSGM mixed score: eps = (1-a)*sigma*x + a*eps'(x, t),
    a = sigmoid(mix) per dimension, i.e. a geometric mixture with the N(0, I) prior.
    """

    def __init__(self, dim, width=512, blocks=4, temb=128, mix_init=None):
        super().__init__()
        self.mix = None if mix_init is None else nn.Parameter(torch.full((dim,), float(mix_init)))
        self.embed = TimeEmbed(temb)
        self.tmlp = nn.Sequential(nn.Linear(temb, temb), nn.SiLU(), nn.Linear(temb, temb))
        self.inp = nn.Linear(dim, width)
        self.blocks = nn.ModuleList(ResBlock(width, temb) for _ in range(blocks))
        self.out = nn.Sequential(nn.LayerNorm(width), nn.SiLU(), nn.Linear(width, dim))

    def forward(self, x, t):
        te = self.tmlp(self.embed(t))
        h = self.inp(x)
        for block in self.blocks:
            h = block(h, te)
        out = self.out(h)
        if self.mix is None:
            return out
        a = torch.sigmoid(self.mix)
        return (1 - a) * marginal(t)[1][:, None] * x + a * out


def score(model, x, t):
    _, sigma = marginal(t)
    return -model(x, t) / sigma[:, None]


# ---------------------------------------------------------------------------- data
def load_run(run: Path, epoch: int):
    with open(run / f"z.{epoch}.pkl", "rb") as fh:
        z = np.asarray(pickle.load(fh), dtype=np.float64)
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    if z.ndim != 2 or z.shape[1] != cfg["model_args"]["zdim"]:
        raise ValueError("Latent shape disagrees with config")
    return z, cfg


def load_model(outdir: Path, device):
    ckpt = torch.load(outdir / "model.pt", map_location=device, weights_only=False)
    model = EpsNet(ckpt["dim"], ckpt["width"], ckpt["blocks"],
                   mix_init=0.0 if ckpt.get("mix") else None).to(device)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    return model, ckpt


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=float) + "\n")


# --------------------------------------------------------------------------- train
def train(args) -> None:
    run = args.run.resolve()
    z, _ = load_run(run, args.epoch)
    out = args.outdir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(z))
    n_hold = int(round(args.holdout * len(z)))
    hold, tr = np.sort(perm[:n_hold]), np.sort(perm[n_hold:])
    # One global scale keeps Euclidean geometry (and hence density modes) intact.
    mean = z[tr].mean(0)
    scale = float(z[tr].std())
    np.save(out / "holdout_rows.npy", hold)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    x_tr = torch.tensor((z[tr] - mean) / scale, dtype=torch.float32, device=device)
    x_ho = torch.tensor((z[hold] - mean) / scale, dtype=torch.float32, device=device)
    model = EpsNet(z.shape[1], args.width, args.blocks).to(device)
    ema = EpsNet(z.shape[1], args.width, args.blocks).to(device)
    ema.load_state_dict(model.state_dict())
    ema.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    warm = 1000
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / warm) * (
        0.5 * (1 + math.cos(math.pi * min(1.0, max(0, i - warm) / max(1, args.iters - warm))))
        * 0.95 + 0.05))
    gen = torch.Generator(device=device).manual_seed(args.seed + 1)
    t_grid = torch.tensor([0.01, 0.05, 0.1, 0.25, 0.5, 0.9], device=device)

    def val_loss(net, x):
        g = torch.Generator(device=device).manual_seed(123)
        losses = []
        with torch.no_grad():
            for t0 in t_grid:
                t = t0.expand(len(x))
                a, s = marginal(t)
                e = torch.randn(x.shape, generator=g, device=device)
                losses.append(((net(a[:, None] * x + s[:, None] * e, t) - e) ** 2).mean().item())
        return losses

    log = []
    t0 = time.time()
    for it in range(args.iters):
        idx = torch.randint(0, len(x_tr), (args.batch,), device=device, generator=gen)
        x0 = x_tr[idx]
        t = torch.rand(args.batch, device=device, generator=gen) * (1 - T_EPS) + T_EPS
        a, s = marginal(t)
        e = torch.randn(x0.shape, device=device, generator=gen)
        loss = ((model(a[:, None] * x0 + s[:, None] * e, t) - e) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - args.ema)
        if (it + 1) % args.log_every == 0 or it == 0:
            sub = x_tr[:len(x_ho)]
            rec = dict(iter=it + 1, loss=loss.item(), lr=sched.get_last_lr()[0],
                       train_val=val_loss(ema, sub), holdout_val=val_loss(ema, x_ho),
                       sec=time.time() - t0)
            log.append(rec)
            LOG.info("it %d loss %.4f holdout(t=.01,.05,.1) %s train %s", it + 1, rec["loss"],
                     np.round(rec["holdout_val"][:3], 4), np.round(rec["train_val"][:3], 4))
    torch.save(dict(ema=ema.state_dict(), model=model.state_dict(), dim=z.shape[1],
                    width=args.width, blocks=args.blocks, mean=mean, scale=scale,
                    run=str(run), epoch=args.epoch, seed=args.seed), out / "model.pt")
    write_json(out / "train.json", dict(
        run=str(run), epoch=args.epoch, n=len(z), n_train=len(tr), n_holdout=len(hold),
        mean=mean.tolist(), scale=scale, sde=dict(type="VP", beta_min=BETA_MIN, beta_max=BETA_MAX,
                                                  t_eps=T_EPS),
        net=dict(width=args.width, blocks=args.blocks, params=sum(p.numel() for p in model.parameters())),
        opt=dict(iters=args.iters, batch=args.batch, lr=args.lr, ema=args.ema, warmup=warm),
        val_t_grid=t_grid.tolist(), log=log))
    LOG.info("Saved %s (%.1f min)", out / "model.pt", (time.time() - t0) / 60)


# ------------------------------------------------------------------------ likelihood
def ode_nll(model, x, steps=400):
    """Exact-divergence probability-flow ODE log-likelihood (nats) in model units."""
    dim = x.shape[1]
    ts = torch.linspace(T_EPS, 1.0, steps + 1, device=x.device)

    def f(x, t):
        tt = t.expand(len(x))
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            drift = -0.5 * beta(tt)[:, None] * (x + score(model, x, tt))
            div = torch.zeros(len(x), device=x.device)
            for d in range(dim):
                div += torch.autograd.grad(drift[:, d].sum(), x, retain_graph=d < dim - 1)[0][:, d]
        return drift.detach(), div.detach()

    logdet = torch.zeros(len(x), device=x.device)
    for i in range(steps):  # Heun
        t, h = ts[i], ts[i + 1] - ts[i]
        d1, v1 = f(x, t)
        xp = x + h * d1
        d2, v2 = f(xp, ts[i + 1])
        x = x + 0.5 * h * (d1 + d2)
        logdet = logdet + 0.5 * h * (v1 + v2)
    prior = -0.5 * (x ** 2).sum(1) - 0.5 * dim * math.log(2 * math.pi)
    return (prior + logdet).cpu().numpy()


def batched(fn, x, bs):
    return np.concatenate([fn(x[i:i + bs]) for i in range(0, len(x), bs)])


# -------------------------------------------------------------------------- sampling
def pc_sample(model, n, dim, device, steps=1000, snr=0.16, seed=0):
    """Reverse-diffusion predictor + one Langevin corrector step (Song et al. 2021)."""
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(n, dim, device=device, generator=g)
    ts = torch.linspace(1.0, T_EPS, steps, device=device)
    dt = (1.0 - T_EPS) / steps
    with torch.no_grad():
        for t0 in ts:
            t = t0.expand(n)
            # Langevin corrector; alpha_i = 1 - beta_i of the DDPM discretisation
            alpha = 1 - beta(t0) * dt
            grad = score(model, x, t)
            noise = torch.randn(x.shape, device=device, generator=g)
            step = 2 * alpha * (snr * noise.norm(dim=1).mean() / grad.norm(dim=1).mean()) ** 2
            x = x + step * grad + torch.sqrt(2 * step) * noise
            # reverse-diffusion predictor (Euler-Maruyama on the reverse SDE)
            b = beta(t)[:, None]
            x_mean = x + (0.5 * b * x + b * score(model, x, t)) * dt
            x = x_mean + torch.sqrt(b * dt) * torch.randn(x.shape, device=device, generator=g)
    return x_mean.cpu().numpy()


# ----------------------------------------------------------------------- fidelity
def c2st(a, b, seed=0):
    """Classifier two-sample test accuracy (0.5 = indistinguishable)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    n = min(len(a), len(b))
    X = np.vstack([a[:n], b[:n]])
    y = np.r_[np.zeros(n), np.ones(n)]
    clf = HistGradientBoostingClassifier(max_iter=300, random_state=seed)
    return float(cross_val_score(clf, X, y, cv=5).mean())


def pc_tvd(ref, other, pcs, bins=100):
    out = []
    for k in range(pcs.shape[0]):
        r, o = ref @ pcs[k], other @ pcs[k]
        edges = np.linspace(*np.quantile(r, [0.0005, 0.9995]), bins + 1)
        hr, _ = np.histogram(np.clip(r, edges[0], edges[-1]), edges)
        ho, _ = np.histogram(np.clip(o, edges[0], edges[-1]), edges)
        out.append(0.5 * np.abs(hr / hr.sum() - ho / ho.sum()).sum())
    return out


# ------------------------------------------------------------------- mode discovery
def ascend(model, x, t0, iters=1500, floor=0.1, bs=65536):
    """Gradient ascent x <- x + c*score on p_t with c = max(sigma^2, floor).

    c = sigma^2 is the Tweedie / mean-shift step; the floor keeps fine scales from
    stalling (curvature of p_t <= 1/(w^2+sigma^2) for local data width w, so c*curv < 2).
    Converges to local maxima of the smoothed density p_t (data convolved with
    N(0, sigma_t^2) after scaling by alpha_t). Returns terminals in z-scaled units (x/alpha).
    """
    t = torch.tensor(t0, device=x.device)
    a, s = marginal(t)
    c = max(s.item() ** 2, floor)
    res, gnorm = [], []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            y = a * x[i:i + bs]
            tt = t.expand(len(y))
            for _ in range(iters):
                y = y - (c / s) * model(y, tt)  # score = -eps / sigma
            gnorm.append((s * model(y, tt)).norm(dim=1) / s)  # |score|*sigma
            res.append(y / a)
    return torch.cat(res), torch.cat(gnorm)


def merge_terminals(term, tol):
    """Group ascent terminals: grid-dedupe then single linkage at distance tol."""
    from scipy.cluster.hierarchy import fcluster, linkage
    key = np.round(term / (tol / 4)).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    reps = np.zeros((len(uniq), term.shape[1]))
    np.add.at(reps, inv, term)
    reps /= np.bincount(inv)[:, None]
    if len(reps) == 1:
        return np.zeros(len(term), dtype=int)
    if len(reps) > 20000:
        raise RuntimeError(f"{len(reps)} unmerged terminals: ascent did not converge")
    lab = fcluster(linkage(reps, "single"), tol, criterion="distance") - 1
    return lab[inv]


def hessian_eigs(model, x, t0):
    t = torch.full((len(x),), t0, device=x.device)
    out = []
    for xi, ti in zip(x, t):
        J = torch.autograd.functional.jacobian(lambda v: score(model, v[None], ti[None])[0], xi)
        out.append(torch.linalg.eigvalsh(0.5 * (J + J.T)).cpu().numpy())
    return np.array(out)


def path_logp(model, xa, xb, t0, n=200):
    """Relative log p_t along the straight segment a->b by line-integrating the score."""
    u = torch.linspace(0, 1, n, device=xa.device)[:, None]
    pts = xa[None] + u * (xb - xa)[None]
    with torch.no_grad():
        s = score(model, pts, torch.full((n,), t0, device=xa.device))
    g = (s @ (xb - xa)).cpu().numpy()
    return np.concatenate([[0], np.cumsum(0.5 * (g[1:] + g[:-1]) / (n - 1))])


def modes_at_scale(model, x_all, t0, min_frac, device):
    alpha, sigma = (v.item() for v in marginal(torch.tensor(t0)))
    term, gn = ascend(model, x_all, t0)
    term = term.cpu().numpy()
    lab = merge_terminals(term, tol=max(0.05, 0.25 * sigma / alpha))
    counts = np.bincount(lab)
    order = np.argsort(-counts)
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    lab = remap[lab]
    counts = counts[order]
    centers = np.array([np.median(term[lab == k], axis=0) for k in range(len(counts))])
    major = np.flatnonzero(counts >= min_frac * len(lab))
    ctr = torch.tensor(centers[major] * alpha, dtype=torch.float32, device=device)
    eig = hessian_eigs(model, ctr, t0) if len(major) else np.zeros((0, x_all.shape[1]))
    barriers = {}
    for i in range(len(major)):
        for j in range(i + 1, len(major)):
            lp = path_logp(model, ctr[i], ctr[j], t0)
            # depth below the lower endpoint (straight path: upper bound on the true barrier)
            barriers[f"{i}-{j}"] = float(min(lp[0], lp[-1]) - lp.min())
    return dict(t=t0, alpha=alpha, sigma=sigma, sigma_over_alpha=sigma / alpha,
                n_terminals=int(len(counts)), n_major=int(len(major)),
                populations=counts.tolist(), major_frac=(counts[major] / len(lab)).tolist(),
                max_hessian_eig=eig.max(1).tolist() if len(major) else [],
                barrier_nats=barriers,
                residual_score_q99=float(torch.quantile(gn.float(), 0.99).item())), lab, centers


def analyze(args) -> None:
    from sklearn.cluster import HDBSCAN, KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.mixture import GaussianMixture

    out = args.outdir.resolve()
    device = torch.device(args.device)
    model, ck = load_model(out, device)
    z, _ = load_run(Path(ck["run"]), ck["epoch"])
    mean, scale = ck["mean"], ck["scale"]
    x = (z - mean) / scale
    dim = x.shape[1]
    hold = np.load(out / "holdout_rows.npy")
    tr = np.setdiff1d(np.arange(len(z)), hold)
    rng = np.random.default_rng(0)
    res = dict(model=str(out / "model.pt"))
    log_jac = -dim * math.log(scale)  # nats: log p_z = log p_x - D log(scale)

    # 1. likelihood: diffusion (PF-ODE) vs Gaussian / BIC-selected GMM, held-out vs train
    tr_sub = rng.choice(tr, len(hold), replace=False)
    f = lambda a: ode_nll(model, torch.tensor(a, dtype=torch.float32, device=device))
    ll_ho = batched(f, x[hold], 16384) + log_jac
    ll_tr = batched(f, x[tr_sub], 16384) + log_jac
    gmm = {}
    fit_rows = rng.choice(tr, min(100000, len(tr)), replace=False)
    for k in (1, 2, 3, 4, 6, 8, 12, 16, 24):
        g = GaussianMixture(k, covariance_type="full", random_state=0, n_init=2).fit(x[fit_rows])
        gmm[k] = dict(bic=float(g.bic(x[fit_rows])),
                      holdout_ll=float(g.score(x[hold]) + log_jac))
    kbest = min(gmm, key=lambda k: gmm[k]["bic"])
    res["likelihood_nats_per_particle"] = dict(
        diffusion_holdout=float(ll_ho.mean()), diffusion_train_subset=float(ll_tr.mean()),
        gaussian_holdout=gmm[1]["holdout_ll"], gmm_bic_best_k=kbest,
        gmm_best_holdout=gmm[kbest]["holdout_ll"], gmm=gmm)
    LOG.info("held-out LL: diffusion %.4f | train %.4f | gauss %.4f | GMM(k=%d) %.4f",
             ll_ho.mean(), ll_tr.mean(), gmm[1]["holdout_ll"], kbest, gmm[kbest]["holdout_ll"])

    # 2. samples + fidelity
    samp = pc_sample(model, args.samples, dim, device, seed=args.seed)
    np.save(out / "samples_z.npy", samp * scale + mean)
    gauss = rng.standard_normal((args.samples, dim))  # VAE N(0,I) prior, in z units
    gauss_x = (gauss - mean) / scale
    pcs = np.linalg.eigh(np.cov(x[tr].T))[1][:, ::-1].T
    ref = x[rng.choice(hold, min(len(hold), args.samples), replace=False)]
    trref = x[rng.choice(tr, len(ref), replace=False)]
    res["fidelity"] = dict(
        c2st_diffusion_vs_holdout=c2st(samp, ref), c2st_train_vs_holdout=c2st(trref, ref),
        c2st_vae_prior_vs_holdout=c2st(gauss_x, ref),
        pc_tvd_diffusion=pc_tvd(ref, samp, pcs), pc_tvd_train=pc_tvd(ref, trref, pcs),
        pc_tvd_vae_prior=pc_tvd(ref, gauss_x, pcs))
    LOG.info("C2ST acc: diffusion %.3f, train-vs-holdout %.3f, N(0,I) prior %.3f",
             *[res["fidelity"][k] for k in ("c2st_diffusion_vs_holdout",
                                            "c2st_train_vs_holdout", "c2st_vae_prior_vs_holdout")])

    # 3. clustering of generated samples (as specified) vs equal-size real z
    sc = {}
    for name, data in (("samples", samp), ("real_z", x[rng.choice(len(x), len(samp), replace=False)])):
        sil = {}
        for k in range(2, 9):
            km = KMeans(k, n_init=4, random_state=0).fit(data)
            sil[k] = float(silhouette_score(data, km.labels_, sample_size=10000, random_state=0))
        hd = HDBSCAN(min_cluster_size=max(50, len(data) // 100), min_samples=25).fit(data[:20000])
        nh = int(hd.labels_.max() + 1)
        sc[name] = dict(kmeans_silhouette=sil, silhouette_argmax=max(sil, key=sil.get),
                        hdbscan_clusters=nh, hdbscan_noise_frac=float((hd.labels_ < 0).mean()))
    res["sample_clustering"] = sc
    LOG.info("sample clustering: %s", sc)

    # 4. scale-space score ascent from every particle
    x_all = torch.tensor(x, dtype=torch.float32, device=device)
    scales, labels = [], {}
    for t0 in args.t_grid:
        info, lab, centers = modes_at_scale(model, x_all, t0, args.min_frac, device)
        info["centers_z"] = (centers[:info["n_major"]] * scale + mean).tolist()
        scales.append(info)
        labels[f"t{t0:g}"] = lab
        LOG.info("t=%.3f sigma/alpha=%.3f: %d terminals, %d major (>=%.1f%%) fracs %s barriers %s",
                 t0, info["sigma_over_alpha"], info["n_terminals"], info["n_major"],
                 100 * args.min_frac, np.round(info["major_frac"], 3),
                 {k: round(v, 2) for k, v in info["barrier_nats"].items()})
    res["scale_space"] = scales
    np.savez_compressed(out / "basin_labels.npz", **labels)
    np.save(out / "loglik_holdout.npy", ll_ho)
    write_json(out / "analysis.json", res)
    LOG.info("Wrote %s", out / "analysis.json")


# --------------------------------------------------------------------------- decode
def decode(args) -> None:
    out = args.outdir.resolve()
    res = json.loads((out / "analysis.json").read_text())
    info = next(s for s in res["scale_space"] if abs(s["t"] - args.t) < 1e-9)
    ck = torch.load(out / "model.pt", map_location="cpu", weights_only=False)
    run = Path(ck["run"])
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    centers = np.asarray(info["centers_z"])
    dest = out / f"modes_t{args.t:g}"
    dest.mkdir(exist_ok=False)
    np.savetxt(dest / "modes_z.txt", centers, fmt="%.10g")
    # nearest observed particle to each mode, for provenance
    z, _ = load_run(run, ck["epoch"])
    near = [int(np.argmin(((z - c) ** 2).sum(1))) for c in centers]
    write_json(dest / "modes.json", dict(t=args.t, populations=info["major_frac"],
                                         nearest_particle_row=near, centers_z=centers.tolist(),
                                         weights=str(run / f"weights.{ck['epoch']}.pkl")))
    box = int(cfg["lattice_args"]["D"]) - 1
    apix = args.apix if args.apix else None
    cmd = [str(Path(sys.executable).with_name("cryodrgn")), "eval_vol",
           str(run / f"weights.{ck['epoch']}.pkl"), "-c", str(run / "config.yaml"),
           "--zfile", str(dest / "modes_z.txt"), "-o", str(dest / "volumes")]
    if apix:
        cmd += ["--Apix", str(apix)]
    if args.gpu is not None:
        cmd += ["--device", str(args.gpu)]
    LOG.info("box %d: %s", box, " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--run", type=Path, required=True)
    t.add_argument("--epoch", type=int, default=100)
    t.add_argument("--outdir", type=Path, required=True)
    t.add_argument("--device", default="cuda:0")
    t.add_argument("--iters", type=int, default=40000)
    t.add_argument("--batch", type=int, default=8192)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--ema", type=float, default=0.999)
    t.add_argument("--width", type=int, default=512)
    t.add_argument("--blocks", type=int, default=4)
    t.add_argument("--holdout", type=float, default=0.05)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--log-every", type=int, default=2000)
    a = sub.add_parser("analyze")
    a.add_argument("--outdir", type=Path, required=True)
    a.add_argument("--device", default="cuda:0")
    a.add_argument("--samples", type=int, default=50000)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--min-frac", type=float, default=0.005)
    a.add_argument("--t-grid", type=float, nargs="+",
                   default=[0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3])
    d = sub.add_parser("decode")
    d.add_argument("--outdir", type=Path, required=True)
    d.add_argument("--t", type=float, required=True)
    d.add_argument("--apix", type=float)
    d.add_argument("--gpu", type=int)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dict(train=train, analyze=analyze, decode=decode)[args.cmd](args)


if __name__ == "__main__":
    main()
