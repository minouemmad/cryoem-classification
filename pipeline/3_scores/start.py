"""(2026-11-25) Alternative starting axes for the fully conditioned pipeline (replace decode + volume-PC scores; only column 0 of pc.scores.npy is used downstream).
  rand OUT            -> OUT/start/rand.scores.npy  random column (fixed seed): makes the first residual model unconditioned (bins = consensus)
  latent MODELDIR EP OUT -> OUT/pc.scores.npy = PCA of the encoder latents z.EP.pkl (no decoder; conditional RAE: free part only)
  free OUT            -> OUT/pc.scores.npy = PCA of the unconditioned residual-model coordinates OUT/start/ppca_null_z.npy (no cryoDRGN at all; 3DVA-like)
  vq | diffmap | mlp OUT -> non-network starts on the same z0: codebook (k-means cells) + back-projected cell maps + map PCA | diffusion-map embedding | MLP autoencoder bottleneck"""
import sys, os, json, pickle, numpy as np
mode = sys.argv[1]; SEED = int(os.environ.get("MP_SEED", "0"))   # (11-25) MP_SEED: replicate seed for random bins / k-means / diffusion subsample / MLP init
if mode == "rand":
    out = sys.argv[2]; p = pickle.load(open(os.environ["MP_POSES"], "rb")); n = len(p[0] if isinstance(p, tuple) else p)
    os.makedirs(out + "/start", exist_ok=True); np.save(out + "/start/rand.scores.npy", np.random.default_rng(SEED).standard_normal((n, 1)).astype(np.float32)); print("random scores", n); sys.exit()
if mode in ("vq", "diffmap", "mlp"):   # (11-25) non-network starts built on the unconditioned residual-model coordinates z0 (OUT/start/ppca_null_z.npy, from the free stage)
    out = sys.argv[2]; z0 = np.load(out + "/start/ppca_null_z.npy").astype(np.float64); Zs = (z0 - z0.mean(0)) / z0.std(0); N = len(Zs); S = np.zeros((N, 10), np.float32); info = dict(mode=mode, n=N, seed=SEED)
    from sklearn.neighbors import NearestNeighbors
    if mode == "vq":      # codebook (k-means cells in z0) -> back-projected cell maps -> weighted PCA of the maps -> particle score = mean over the nearest cells
        from sklearn.cluster import MiniBatchKMeans
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../lib")); from fastbp import FastBP, lowpass
        NC = int(os.environ.get("MP_VQ_NC", "300")); km = MiniBatchKMeans(NC, random_state=SEED, n_init=3, batch_size=4096).fit(Zs); c = km.labels_; n = np.bincount(c, minlength=NC).astype(float)
        bp = FastBP(os.environ["MP_STACK"], os.environ["MP_POSES"], os.environ["MP_CTF"]); ap = bp.apix * 2; env = np.load(os.environ["MP_ENV64"]).astype(bool)
        def crop(v):
            f = np.fft.fftshift(np.fft.fftn(v)); f = f[32:96, 32:96, 32:96]; return lowpass(np.real(np.fft.ifftn(np.fft.ifftshift(f))).astype(np.float32) / 8, ap, 10.0)[env]
        F = []
        for k in range(NC):
            a, b = bp.halves(np.flatnonzero(c == k), seed=0); F.append((crop(a) + crop(b)) / 2)
        F = np.array(F, np.float64); wn = n / n.sum(); Fc = F - wn @ F; sw = np.sqrt(wn)[:, None]
        lam, u = np.linalg.eigh((sw * Fc) @ (sw * Fc).T); o = np.argsort(-lam)[:10]; lam, u = lam[o], u[:, o]
        V = (sw * Fc).T @ u; V /= np.linalg.norm(V, axis=0); P = Fc @ V; tot = float((wn[:, None] * Fc ** 2).sum())
        kk = int(os.environ.get("MP_VQ_KNN", "8")); idx = NearestNeighbors(n_neighbors=kk).fit(km.cluster_centers_).kneighbors(Zs, return_distance=False); S[:] = P[idx].mean(1)
        info.update(cells=NC, knn=kk, var=[float(l / tot) for l in lam]); np.save(out + "/start/vq_cells.npy", c.astype(np.int16)); np.save(out + "/start/vq_cellmaps.npy", F.astype(np.float32))
    elif mode == "diffmap":   # diffusion-map / spectral embedding on a subsample, extended to all particles by k-NN regression
        from sklearn.manifold import SpectralEmbedding
        from sklearn.neighbors import KNeighborsRegressor
        sub = np.random.default_rng(SEED).choice(N, min(N, int(os.environ.get("MP_DM_SUB", "30000"))), replace=False)
        E = SpectralEmbedding(n_components=10, affinity="nearest_neighbors", n_neighbors=int(os.environ.get("MP_DM_K", "30")), random_state=SEED).fit_transform(Zs[sub])
        S[:] = KNeighborsRegressor(n_neighbors=15, weights="distance").fit(Zs[sub], E).predict(Zs); info.update(sub=len(sub))
    else:                     # nonlinear PCA: denoising MLP autoencoder 8 -> 128 -> 128 -> 1 -> 128 -> 128 -> 8; start = the bottleneck (a principal curve)
        import torch, torch.nn as nn
        torch.manual_seed(SEED); dev = "cuda" if torch.cuda.is_available() else "cpu"; d = Zs.shape[1]; X = torch.tensor(Zs, dtype=torch.float32, device=dev)
        enc = nn.Sequential(nn.Linear(d, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 1)).to(dev)
        dec = nn.Sequential(nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, d)).to(dev)
        opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), 1e-3); EP = int(os.environ.get("MP_MLP_EP", "40")); losses = []
        for ep in range(EP):
            perm = torch.randperm(N, device=dev); tot = 0.0
            for i in range(0, N, 4096):
                xb = X[perm[i:i + 4096]]; loss = ((dec(enc(xb + 0.1 * torch.randn_like(xb))) - xb) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(xb)
            losses.append(tot / N)
        with torch.no_grad(): h = torch.cat([enc(X[i:i + 65536]) for i in range(0, N, 65536)]).cpu().numpy()[:, 0]
        lin = np.linalg.svd(Zs - Zs.mean(0), full_matrices=False)[2][0]; S[:, 0] = h
        info.update(epochs=EP, loss_first=losses[0], loss_last=losses[-1], var_explained=float(1 - losses[-1]), corr_with_linear_pc1=float(abs(np.corrcoef(h, Zs @ lin)[0, 1])))
    S[:, 0] = (S[:, 0] - S[:, 0].mean()) / S[:, 0].std(); np.save(out + "/pc.scores.npy", S); json.dump(info, open(out + "/start.json", "w"), indent=1); print(json.dumps(info)); sys.exit()
if mode == "latent":
    md, ep, out = sys.argv[2:5]; z = pickle.load(open(f"{md}/z.{ep}.pkl", "rb")); z = z[0] if isinstance(z, tuple) else z; src = f"{md}/z.{ep}.pkl"
    if os.path.exists(md + "/cond.json"): z = np.asarray(z)[:, : json.load(open(md + "/cond.json"))["zfree"]]; src += " (free part of a conditional RAE)"
else:
    out = sys.argv[2]; src = out + "/start/ppca_null_z.npy"; z = np.load(src)
z = np.asarray(z, np.float64); z -= z.mean(0); lam, V = np.linalg.eigh(z.T @ z / len(z)); o = np.argsort(-lam); lam, V = lam[o], V[:, o]; P = z @ V
S = np.zeros((len(z), 10), np.float32); k = min(10, P.shape[1]); S[:, :k] = P[:, :k]; np.save(out + "/pc.scores.npy", S)
var = (lam / lam.sum()).tolist(); json.dump(dict(mode=mode, source=src, n=len(z), var=var), open(out + "/start.json", "w"), indent=1)
print(mode, src, len(z), "particles; variance fraction per component:", " ".join(f"{v:.3f}" for v in var))
