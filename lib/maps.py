"""Shared map helpers: back-projected (half) maps at a fixed resolution inside the protein envelope, and class-map algebra."""
import os
import numpy as np
from fastbp import lowpass


def up2(m):
    """64^3 -> 128^3 by voxel repetition (envelopes / masks)"""
    return np.repeat(np.repeat(np.repeat(m, 2, 0), 2, 1), 2, 2)


def cropper(env64, apix64):
    """crop(v): 128^3 back-projection -> voxels inside the envelope, low-passed to MP_LP A (default 10),
    on the 64^3 grid (Fourier crop, default) or the full 128^3 grid (MP_GRID=128)."""
    lp, grid, env = float(os.environ.get("MP_LP", "10")), int(os.environ.get("MP_GRID", "64")), env64.astype(bool)
    if grid == 128:
        env = up2(env)
        return lambda v: lowpass(np.asarray(v, dtype=np.float32), apix64 / 2, lp)[env]

    def crop(v):
        f = np.fft.fftshift(np.fft.fftn(v))[32:96, 32:96, 32:96]
        return lowpass(np.real(np.fft.ifftn(np.fft.ifftshift(f))).astype(np.float32) / 8, apix64, lp)[env]
    return crop


def group_halves(bp, labels, K, crop, seed=0):
    """half maps A, B (K x voxels, float64) of the particle groups labels == 0..K-1"""
    A, B = [], []
    for k in range(K):
        a, b = bp.halves(np.flatnonzero(labels == k), seed=seed); A.append(crop(a)); B.append(crop(b))
    return np.array(A, np.float64), np.array(B, np.float64)


cc = lambda a, b: float(np.corrcoef(a, b)[0, 1])


def class_maps(X, n, lab, K):
    """n-weighted mean of the cell maps X (cells x voxels) in each class"""
    return np.array([(X[lab == k] * n[lab == k, None]).sum(0) / n[lab == k].sum() for k in range(K)])


def own_component(M, k):
    """part of class map M[k] that is not an affine blend of the other class maps (residual after projection on their span)"""
    O = np.delete(M, k, 0); base = O[0]; Bm = (O[1:] - base).T; h = M[k] - base
    return h - (Bm @ np.linalg.lstsq(Bm, h, rcond=None)[0] if Bm.shape[1] else 0)
