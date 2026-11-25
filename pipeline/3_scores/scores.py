#!/usr/bin/env python
"""Label-free volume-PC scores of every particle for one decoded model.
Medoid maps -> Gaussian-smoothed (MP_SCFW A FWHM, default 8) correlation features inside MASK64 -> particle-weighted PCA (weight = particles
nearest each medoid) -> each particle scored by the mean PC value of its k nearest medoids in latent space (k by leave-one-out on the medoids).
Usage: scores.py DECODED MASK64 OUT_PREFIX [NPC=10]   -> OUT_PREFIX.scores.npy (N x NPC), OUT_PREFIX.json (variance fraction per PC)"""
import json, os, sys
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
from cryo_utils import read_volume, load_latents, volume_features

dec, mask64, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]; NPC = int(sys.argv[4]) if len(sys.argv) > 4 else 10
meta = json.loads((dec / "sample.json").read_text())
z = load_latents(Path(meta["run"]) / f"z.{meta['epoch']}.pkl"); C = np.loadtxt(dec / "centers.txt")
paths = [dec / "volumes" / f"vol_{i + 1:03d}.mrc" for i in range(meta["samples"])]
F = volume_features(paths, read_volume(mask64, meta["box"], meta["apix"]), meta["box"], meta["apix"], float(os.environ.get("MP_SCFW", "8.0")), "correlation", 32).astype(np.float64)
near = NearestNeighbors(n_neighbors=1).fit(C).kneighbors(z, return_distance=False)[:, 0]
wn = np.bincount(near, minlength=len(C)) + 1.0; wn /= wn.sum(); Fc = F - wn @ F; s = np.sqrt(wn)[:, None]
lam, u = eigsh((s * Fc) @ (s * Fc).T, k=NPC, which="LA"); o = np.argsort(-lam); lam, u = lam[o], u[:, o]
V = (s * Fc).T @ u; V /= np.linalg.norm(V, axis=0); P = Fc @ V; tot = float((wn[:, None] * Fc ** 2).sum())


def choose_k(proj, ks=(1, 2, 4, 8, 16, 32)):
    """number of neighbouring medoids whose mean best predicts a medoid's own PC value (leave-one-out)"""
    idx = NearestNeighbors(n_neighbors=min(max(ks) + 1, len(C))).fit(C).kneighbors(C, return_distance=False)[:, 1:]
    return min((k for k in ks if k < idx.shape[1] + 1), key=lambda k: np.mean((proj[idx[:, :k]].mean(1) - proj) ** 2))


S = np.stack([P[:, k][NearestNeighbors(n_neighbors=choose_k(P[:, k])).fit(C).kneighbors(z, return_distance=False)].mean(1) for k in range(NPC)], 1)
np.save(out + ".scores.npy", S.astype(np.float32)); var = [float(l / tot) for l in lam]
json.dump(dict(decoded=str(dec), n=int(len(z)), var=var), open(out + ".json", "w"), indent=1)
print(f"{len(z)} particles, {len(C)} medoids; variance fraction per volume PC: " + " ".join(f"{v:.3f}" for v in var))
