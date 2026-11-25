"""K = 20 bins along the dominant latent PC (PCA of z) and the dominant volume PC (weighted PCA of the decoded
centres; particle score = nearest centre's) -> bins_*.npy, z_*.txt (mean z per bin), scores_*.npy, stats.json.
Usage: prepare.py NAME RUN_DIR DECODED_DIR MASK64 OUT_DIR
"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cryo_utils import load_latents, read_volume, volume_features  # noqa: E402
from scipy.sparse.linalg import eigsh  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

name, run, dec, mask64, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], Path(sys.argv[5])
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
z = load_latents(run / "z.50.pkl")
zm = z.mean(0); zc = z - zm
lam, U = np.linalg.eigh(np.cov(zc.T)); o = np.argsort(-lam); lam, U = lam[o], U[:, o]
L = zc @ U
boot = []
for _ in range(50):
    s = zc[rng.integers(0, len(z), len(z))]
    w, v = np.linalg.eigh(np.cov(s.T)); boot.append(abs(float(v[:, -1] @ U[:, 0])))

meta = json.loads((dec / "sample.json").read_text())
C = np.loadtxt(dec / "centers.txt")
near = NearestNeighbors(n_neighbors=1).fit(C).kneighbors(z, return_distance=False)[:, 0]
w = np.bincount(near, minlength=len(C)) + 1.0; wn = w / w.sum()
paths = [dec / "volumes" / f"vol_{i + 1:03d}.mrc" for i in range(meta["samples"])]
mask = read_volume(mask64, meta["box"], meta["apix"])
F = volume_features(paths, mask, meta["box"], meta["apix"], 8.0, "correlation", 32).astype(np.float64)
mu = wn @ F; Fc = F - mu; sw = np.sqrt(wn)
G = (sw[:, None] * Fc) @ (sw[:, None] * Fc).T
lam2, u = eigsh(G, k=10, which="LA"); oo = np.argsort(-lam2); lam2, u = lam2[oo], u[:, oo]
V = (sw[:, None] * Fc).T @ u; V /= np.linalg.norm(V, axis=0)
P = Fc @ V
vboot = []
for _ in range(50):                                   # resample particles -> centre weights -> PC1 direction
    wb = np.bincount(near[rng.integers(0, len(z), len(z))], minlength=len(C)) + 1.0; wb /= wb.sum()
    Fb = F - wb @ F; sb = np.sqrt(wb)
    _, ub = eigsh((sb[:, None] * Fb) @ (sb[:, None] * Fb).T, k=1, which="LA")
    vb = (sb[:, None] * Fb).T @ ub[:, 0]; vboot.append(abs(float(vb @ V[:, 0] / np.linalg.norm(vb))))
S = P[near, 0]
st = dict(name=name, n=len(z), latent_var=(lam / lam.sum()).tolist(), volume_var=(lam2 / np.trace(G)).tolist(),
          latent_pc1_boot_cos=dict(median=float(np.median(boot)), p05=float(np.percentile(boot, 5))),
          volume_pc1_boot_cos=dict(median=float(np.median(vboot)), p05=float(np.percentile(vboot, 5))), traj={})
for nm, x in (("lat", L[:, 0]), ("vol", S)):
    km = KMeans(20, n_init=10, random_state=0).fit(x[:, None])
    oo = np.argsort(km.cluster_centers_[:, 0]); rank = np.empty(20, int); rank[oo] = np.arange(20)
    lab = rank[km.labels_]
    np.save(out / f"bins_{nm}.npy", lab.astype(np.int8)); np.save(out / f"scores_{nm}.npy", x.astype(np.float32))
    np.savetxt(out / f"z_{nm}.txt", np.array([z[lab == k].mean(0) for k in range(20)]))
    st["traj"][nm] = dict(sizes=np.bincount(lab, minlength=20).tolist(), centres=km.cluster_centers_[oo, 0].tolist())
json.dump(st, open(out / "stats.json", "w"), indent=1)
print(name, "latent var", np.round(st["latent_var"][:5], 3), "boot", st["latent_pc1_boot_cos"])
print(name, "volume var", np.round(st["volume_var"][:5], 3), "boot", st["volume_pc1_boot_cos"])
