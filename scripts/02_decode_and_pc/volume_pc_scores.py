#!/usr/bin/env python
"""Label-free volume-PC scores of every particle for one decoded cryoDRGN run.
Usage: volume_pc_scores.py DECODED MASK64 OUT_PREFIX [NPC] [LABELS.npy]"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from cryo_utils import read_volume, load_latents, volume_features
from recon_tree import choose_k
dec, mask64, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
NPC = int(sys.argv[4]) if len(sys.argv) > 4 else 10
meta = json.loads((dec / "sample.json").read_text())
z = load_latents(Path(meta["run"]) / f"z.{meta['epoch']}.pkl"); C = np.loadtxt(dec / "centers.txt")
paths = [dec / "volumes" / f"vol_{i + 1:03d}.mrc" for i in range(meta["samples"])]
F = volume_features(paths, read_volume(mask64, meta["box"], meta["apix"]), meta["box"], meta["apix"], 8.0, "correlation", 32).astype(np.float64)
near = NearestNeighbors(n_neighbors=1).fit(C).kneighbors(z, return_distance=False)[:, 0]
w = np.bincount(near, minlength=len(C)) + 1.0; wn = w / w.sum(); mu = wn @ F; Fc = F - mu; s = np.sqrt(wn)
G = (s[:, None] * Fc) @ (s[:, None] * Fc).T; lam, u = eigsh(G, k=NPC, which="LA"); order = np.argsort(-lam); lam, u = lam[order], u[:, order]
V = (s[:, None] * Fc).T @ u; V /= np.linalg.norm(V, axis=0); P = Fc @ V; tot = float((wn[:, None] * Fc ** 2).sum())
def score(cp):
    ks, _ = choose_k(C, cp); idx = NearestNeighbors(n_neighbors=ks).fit(C).kneighbors(z, return_distance=False); return cp[idx].mean(1)
S = np.stack([score(P[:, k]) for k in range(NPC)], 1); np.save(out + ".scores.npy", S.astype(np.float32))
var = [float(l / tot) for l in lam]; json.dump(dict(decoded=str(dec), n=int(len(z)), var=var), open(out + ".json", "w"), indent=1)
print(f"{len(z)} particles, {len(C)} centres; variance fraction per volume PC: " + " ".join(f"{v:.3f}" for v in var))
if len(sys.argv) > 5:
    lab = np.load(sys.argv[5]); print("labels:", lab.shape, dict(zip(*[a.tolist() for a in np.unique(lab, return_counts=True)])))
    if lab.shape[0] == len(z):
        keep = lab >= 0
        for k in range(NPC):
            x = S[keep, k]; c = lab[keep]; eta2 = sum((c == g).sum() * (x[c == g].mean() - x.mean()) ** 2 for g in np.unique(c)) / ((x - x.mean()) ** 2).sum()
            print(f"pc{k + 1}  var {var[k]:.3f}  eta2 {eta2:.3f}  " + "  ".join(f"class{g}:{x[c == g].mean() / x.std():+.2f}" for g in np.unique(c)))
