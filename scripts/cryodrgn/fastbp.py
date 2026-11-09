#!/usr/bin/env python
"""In-memory homogeneous backprojection of many particle subsets (consensus poses + CTF).

Same maths as `cryodrgn backproject_voxel` (reuses its add_slice / regularize_volume), but the stack is loaded once,
so each subset costs only the GPU scatter. Half maps use a seeded random 50/50 split.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from cryodrgn import ctf, dataset
from cryodrgn.commands.backproject_voxel import add_slice, regularize_volume
from cryodrgn.lattice import Lattice
from cryodrgn.pose import PoseTracker


class FastBP:
    def __init__(self, particles, poses, ctf_pkl, device="cuda", batch=2000):
        t = time.time()
        self.dev = torch.device(device)
        self.data = dataset.ImageDataset(mrcfile=particles, norm=(0, 1), invert_data=True,
                                         lazy=False, device=self.dev)
        self.D = self.data.D
        self.lat = Lattice(self.D, extent=self.D // 2, device=self.dev)
        self.pose = PoseTracker.load(poses, self.data.N, self.D, None, None, device=self.dev)
        cp = ctf.load_ctf_for_training(self.D - 1, ctf_pkl)
        self.ctf = torch.tensor(cp, device=self.dev)
        self.apix = float(cp[0, 0])
        self.mask = self.lat.get_circular_mask(self.D // 2)
        self.batch = batch
        print(f"FastBP: {self.data.N} images, D={self.D - 1}, apix {self.apix:.4f}, "
              f"loaded in {time.time() - t:.0f}s", flush=True)

    def _sums(self, idx):
        D = self.D
        vol = torch.zeros((D, D, D), device=self.dev)
        cnt = torch.zeros((D, D, D), device=self.dev)
        for s in range(0, len(idx), self.batch):
            ii = np.sort(idx[s: s + self.batch])
            B = len(ii)
            ff, rot, c = self._batch(ii)
            add_slice(vol, cnt, self.lat.coords[self.mask] @ rot, ff, D, c)
        return vol, cnt

    def _batch(self, ii):
        """Centred masked Hartley images (B, M), rotations, CTFs (B, M) for sorted indices ii. If self.sim is set (simulate()),
        the images come from the simulated stack instead of the data."""
        B = len(ii)
        rot, trans = self.pose.get_pose(ii)
        rot = rot.to(self.dev)
        freqs = self.lat.freqs2d.unsqueeze(0).expand(B, *self.lat.freqs2d.shape) / self.ctf[ii, 0].view(B, 1, 1)
        c = ctf.compute_ctf(freqs, *torch.split(self.ctf[ii, 1:], 1, 1)).view(B, -1)[:, self.mask]
        if getattr(self, "sim", None) is not None:
            return self.sim[torch.as_tensor(ii, device=self.dev)].float(), rot, c
        ff = self.data[torch.as_tensor(ii)]["y"].to(self.dev).view(B, -1)[:, self.mask]
        if trans is not None:
            ff = self.lat.translate_ht(ff, trans.to(self.dev).unsqueeze(1), self.mask).view(B, -1)
        return ff, rot, c

    @torch.no_grad()
    def project(self, H, ii, rot):
        """Central slices (B, M) of the Hartley map H at rotations rot."""
        import torch.nn.functional as Fnn
        D, d2 = self.D, self.D // 2
        xyz = self.lat.coords[self.mask] @ rot
        g = ((xyz.clamp(-d2, d2) + d2) / (D - 1) * 2 - 1).unsqueeze(0).unsqueeze(0)
        return Fnn.grid_sample(H[None, None], g, mode="bilinear", align_corners=True).view(len(ii), -1)

    @torch.no_grad()
    def simulate(self, seed=0, batch=2000):
        """Homogeneous synthetic stack (null for classification tests): image_i = a_i * CTF_i * slice(consensus, pose_i) + noise,
        a_i = per-image least-squares scale of the real image to that prediction; noise Gaussian with the per-shell variance
        of the real residuals (x - a c P). Same poses, CTFs, scales and noise spectrum as the data, no heterogeneity."""
        self.sim = None
        N = self.data.N
        H = self.hvol(np.arange(N))
        r = self.lat.freqs2d[self.mask].norm(dim=1) * self.D
        shell = r.round().long().clamp(max=self.D // 2); nsh = int(shell.max()) + 1
        acc = torch.zeros(nsh, device=self.dev, dtype=torch.float64); cnt = torch.zeros(nsh, device=self.dev, dtype=torch.float64)
        scale = torch.empty(N, device=self.dev)
        for s in range(0, N, batch):
            ii = np.arange(s, min(N, s + batch))
            ff, rot, c = self._batch(ii)
            pred = c * self.project(H, ii, rot)
            a = (ff * pred).sum(1) / pred.pow(2).sum(1).clamp(min=1e-20); scale[s: s + len(ii)] = a
            res = ff - a[:, None] * pred
            acc.index_add_(0, shell, res.pow(2).sum(0).double()); cnt.index_add_(0, shell, torch.full_like(shell, len(ii), dtype=torch.float64))
        sd = (acc / cnt.clamp(min=1)).sqrt().float()[shell]
        g = torch.Generator(device=self.dev).manual_seed(seed)
        sim = torch.empty((N, int(self.mask.sum())), device=self.dev, dtype=torch.float16)
        for s in range(0, N, batch):
            ii = np.arange(s, min(N, s + batch))
            _, rot, c = self._batch(ii)
            pred = c * self.project(H, ii, rot)
            sim[s: s + len(ii)] = (scale[s: s + len(ii), None] * pred + sd * torch.randn(pred.shape, device=self.dev, generator=g)).half()
        del H
        self.sim = sim
        print(f"simulated homogeneous stack: {N} images, mean scale {scale.mean().item():.3f}, signal/noise power "
              f"{(scale.pow(2).mean()).item():.3g}", flush=True)

    def halves(self, idx, seed=0):
        """Half maps A, B (numpy, real space) from a seeded random split of idx."""
        idx = np.asarray(idx)
        r = np.random.default_rng(seed).random(len(idx)) < 0.5
        out = []
        for part in (idx[r], idx[~r]):
            v, c = self._sums(part)
            c[c == 0] = 1
            out.append(np.asarray(regularize_volume(v, c, 1.0)).astype(np.float32))
        return out

    # ---- cross-validated image fit (recon_tree --accept cv) ----
    def hvol(self, idx, reg=1.0):
        """Regularised map of the particles idx in the Hartley domain (centred (D,D,D) grid, on device): the same
        estimate as halves() before the inverse transform, kept for projecting central slices."""
        v, c = self._sums(np.asarray(idx))
        c[c == 0] = 1
        rc = c + reg * c.mean()
        rc *= c.mean() / rc.mean()
        return v / rc

    def band_mask(self, lo, hi):
        """Boolean over the masked 2-D lattice pixels: resolution between lo and hi Angstrom."""
        f = self.lat.freqs2d[self.mask].norm(dim=1) / self.apix
        with torch.no_grad():
            res = 1.0 / f.clamp(min=1e-9)
        return (res <= lo) & (res >= hi)

    @torch.no_grad()
    def fit(self, idx, maps, labels, band, batch=1000):
        """Normalised projection score of each image against one map: s_i = <x_i, c_i P_i> / ||c_i P_i|| over the band
        pixels, x_i = centred image (Hartley), c_i = its CTF, P_i = central slice of maps[labels_i] at the image's pose.
        Higher = the map explains the image better (scale-free in the map). Returns numpy (len(idx),) in idx order."""
        import torch.nn.functional as Fnn
        idx = np.asarray(idx); labels = np.asarray(labels)
        order = np.argsort(idx)
        out = np.empty(len(idx), np.float64)
        D, d2 = self.D, self.D // 2
        coords = self.lat.coords[self.mask]
        for s in range(0, len(idx), batch):
            sel = order[s: s + batch]
            ii = idx[sel]; B = len(ii)
            ff, rot, c = self._batch(ii)
            P = torch.empty_like(ff)
            lb = labels[sel]
            for m in np.unique(lb):
                rr = np.flatnonzero(lb == m)
                xyz = coords @ rot[rr]                                        # (b, M, 3), voxel units, x first
                g = ((xyz.clamp(-d2, d2) + d2) / (D - 1) * 2 - 1).unsqueeze(0).unsqueeze(0)
                P[rr] = Fnn.grid_sample(maps[m][None, None], g, mode="bilinear", align_corners=True).view(len(rr), -1)
            pred = (c * P)[:, band]
            out[sel] = ((ff[:, band] * pred).sum(1) / pred.pow(2).sum(1).clamp(min=1e-20).sqrt()).double().cpu().numpy()
        return out


def fsc_curve(a, b, mask=None):
    if mask is not None:
        a, b = a * mask, b * mask
    A, B = np.fft.fftn(a), np.fft.fftn(b)
    n = a.shape[0]
    f = np.fft.fftfreq(n)
    r = np.sqrt(sum(x ** 2 for x in np.meshgrid(f, f, f, indexing="ij")))
    sh = np.round(r * n).astype(int).ravel()
    num = np.bincount(sh, (A * B.conj()).real.ravel())
    den = np.sqrt(np.bincount(sh, (np.abs(A) ** 2).ravel()) * np.bincount(sh, (np.abs(B) ** 2).ravel()))
    return (num / np.maximum(den, 1e-30))[: n // 2]


def resolution(curve, n, apix, thr=0.143):
    below = np.flatnonzero(curve[1:] < thr)
    i = below[0] + 1 if len(below) else len(curve) - 1
    y0, y1 = curve[i - 1], curve[i]
    s = (i - 1) + (y0 - thr) / (y0 - y1) if y0 != y1 else i
    return float(n * apix / s)


def lowpass(v, apix, res):
    n = v.shape[0]
    f = np.fft.fftfreq(n, d=apix)
    r = np.sqrt(sum(x ** 2 for x in np.meshgrid(f, f, f, indexing="ij")))
    filt = np.exp(-0.5 * (r * res / 1.0) ** 2 * 4)  # Gaussian, ~ half amplitude at 1/res
    return np.real(np.fft.ifftn(np.fft.fftn(v) * filt)).astype(np.float32)


def fsc_band(curve, n, apix, lo=10.0, hi=3.0):
    """Mean FSC over shells between lo and hi Angstrom: a smooth alternative to the 0.143 crossing,
    which jumps when the curve plateaus near the threshold."""
    s = np.arange(len(curve))
    with np.errstate(divide="ignore"):
        res = n * apix / s
    sel = (res <= lo) & (res >= hi)
    return float(curve[sel].mean())
