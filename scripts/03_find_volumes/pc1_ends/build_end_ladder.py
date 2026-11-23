"""(1) END LADDER: nested hard windows anchored at each end of the main axis (sizes 10k ... 110k), no class and no size assumed - the end map will be chosen from the refined ladder by the stability rule """
import hashlib, json
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
A = Path('/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn/J4001_gP25_WT/candidate_groups_epoch50/axis_survey')
S0 = np.load(A / 'pc_scan_J4001_rae_nuposes.scores.npy'); N = len(S0)
MODELS = dict(rae=A / 'pc_scan_J4001_rae_nuposes.scores.npy', vae=A / 'pc_scan_J4001_128_nuposes_seed31337_e50.scores.npy')
SIZES = [10000, 15000, 22000, 33000, 50000, 75000, 110000]
def load(m):
    S = np.load(MODELS[m]); sg = float(np.sign(np.corrcoef(S[:, 0], S0[:, 0])[0, 1])); S = S * np.array([sg] + [1.0] * (S.shape[1] - 1))     # only PC1 sign aligned
    order = np.argsort(S[:, 0], kind='stable'); rank = np.empty(N); rank[order] = np.arange(N) / N; return S, order, rank, sg
def write(name, K, groups, info, meta):
    d = A / name / 'groups'; d.mkdir(parents=True, exist_ok=True); wins = []
    for j, (g, inf) in enumerate(zip(groups, info)):
        g = np.sort(np.asarray(g)); np.save(d / f'leaves_K{K}_g{j}.npy', g); wins.append(dict(j=j, K=K, n=int(len(g)), **inf)); print(name, f'g{j}', wins[-1], flush=True)
    json.dump(dict(meta, windows=wins), open(A / name / 'windows.json', 'w'), indent=1)
for m in ('rae', 'vae'):
    S, order, rank, sg = load(m); meta = dict(model=m, scores=MODELS[m].name, pc1_sign=sg, md5=hashlib.md5(open(MODELS[m], 'rb').read()).hexdigest(), N=N)
    G = [order[:n] for n in SIZES] + [order[::-1][:n] for n in SIZES]; I = [dict(end='P8 side (rank 0)', n_target=n) for n in SIZES] + [dict(end='P6 side (rank 1)', n_target=n) for n in SIZES]
    write(f'set_ends_{m}', len(G), G, I, dict(meta, kind='nested end windows'))
    if m != 'vae': continue
    def sd_for(c, n):
        lo, hi = 0.003, 3.0
        for _ in range(70):
            mid = 0.5 * (lo + hi); w = np.exp(-0.5 * ((rank - c) / mid) ** 2); lo, hi = (mid, hi) if w.sum() < n else (lo, mid)
        return 0.5 * (lo + hi)
    rg = np.random.default_rng(21); spec = [(0.40, 43000), (0.445, 43000), (0.50, 43000), (0.43, 58000)]; G = []; I = []
    for c, n in spec:
        sd = sd_for(c, n); G.append(np.where(rg.random(N) < np.exp(-0.5 * ((rank - c) / sd) ** 2))[0]); I.append(dict(centre=c, n_target=n, sd_rank=round(float(sd), 4)))
    write('set_vae_win', 4, G, I, dict(meta, kind='Gaussian soft windows (VAE)'))
    lab = KMeans(4, n_init=10, random_state=0).fit_predict(S[:, :1]); ords = np.argsort([S[lab == c, 0].mean() for c in range(4)])
    write('set_vae_km', 4, [np.where(lab == c)[0] for c in ords], [dict(cluster_rank=i) for i in range(4)], dict(meta, kind='k-means K4 on PC1 (VAE), clusters ordered P8 -> P6'))
    B = 200; idx = np.minimum((rank * B).astype(int), B - 1); C = np.array([S[idx == b].mean(0) for b in range(B)]); k = np.exp(-0.5 * (np.arange(-10, 11) / 3.0) ** 2); k /= k.sum()
    Cs = np.array([np.convolve(np.pad(C[:, d], 10, mode='edge'), k, mode='valid') for d in range(S.shape[1])]).T
    L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Cs, axis=0), axis=1))]); L /= L[-1]; rb = (np.arange(B) + 0.5) / B; K = 4
    cuts = [0.0] + [float(np.interp(i / K, L, rb)) for i in range(1, K)] + [1.0 + 1e-9]
    write('set_vae_arc', K, [np.where((rank >= cuts[i]) & (rank < cuts[i + 1]))[0] for i in range(K)], [dict(cut_lo=round(cuts[i], 4), cut_hi=round(min(cuts[i + 1], 1.0), 4)) for i in range(K)], dict(meta, kind='equal arc length partition K4 (VAE)'))
