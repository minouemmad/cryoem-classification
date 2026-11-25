"""2026-11-15 (user: drop the hard 60k particle count) - four label-free ways to let the window size vary along the RAE volume-PC1."""
import hashlib, json
from pathlib import Path
import numpy as np
A = Path('/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn/J4001_gP25_WT/candidate_groups_epoch50/axis_survey')
SF = A / 'pc_scan_J4001_rae_nuposes.scores.npy'
S = np.load(SF); N = len(S); s = S[:, 0]
order = np.argsort(s, kind='stable'); rank = np.empty(N); rank[order] = np.arange(N) / N
CENTRE = 0.30 + 0.5 * 60000 / N
meta = dict(scores_file='candidate_groups_epoch50/axis_survey/' + SF.name, scores_md5=hashlib.md5(open(SF, 'rb').read()).hexdigest(), N=N)


def write(name, K, groups, info, extra):
    d = A / name / 'groups'; d.mkdir(parents=True, exist_ok=True)
    wins = []
    for j, (g, inf) in enumerate(zip(groups, info)):
        g = np.sort(np.asarray(g)); np.save(d / f'leaves_K{K}_g{j}.npy', g)
        wins.append(dict(j=j, K=K, n=int(len(g)), rank_lo=round(float(rank[g].min()), 4), rank_hi=round(float(rank[g].max()), 4),
                         rank_mean=round(float(rank[g].mean()), 4), pc1_lo=round(float(s[g].min()), 4), pc1_hi=round(float(s[g].max()), 4), **inf))
        print(name, f'K{K}_g{j}', wins[-1], flush=True)
    f = A / name / 'windows.json'; old = json.load(open(f))['windows'] if f.exists() else []
    json.dump(dict(meta, **extra, windows=[w for w in old if w['K'] != K] + wins), open(f, 'w'), indent=1)


# 1 variable width, hard, centred on the rule's pick
ns = [20000, 30000, 40000, 50000, 80000, 100000]; G = []
for n in ns:
    st = int(round(CENTRE * N - n / 2)); G.append(order[st:st + n])
write('set_win_vw', len(ns), G, [dict(n_target=n) for n in ns], dict(kind='hard window, variable n', centre_rank=CENTRE))

# 2 fixed PC1-value width
w = float(np.quantile(s, 0.30 + 60000 / N) - np.quantile(s, 0.30)); lo, hi = np.quantile(s, [0.01, 0.99])
starts = np.linspace(lo, hi - w, 10)
write('set_win_val', 10, [np.where((s >= a) & (s < a + w))[0] for a in starts], [dict(value_start=round(float(a), 4)) for a in starts],
      dict(kind='fixed PC1-value width', value_width=w))

# 3 equal-arc-length partitions of the conditional-mean curve
B = 200; idx = np.minimum((rank * B).astype(int), B - 1); C = np.array([S[idx == b].mean(0) for b in range(B)])
k = np.exp(-0.5 * (np.arange(-10, 11) / 3.0) ** 2); k /= k.sum()
Cs = np.array([np.convolve(np.pad(C[:, d], 10, mode='edge'), k, mode='valid') for d in range(S.shape[1])]).T
L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Cs, axis=0), axis=1))]); L /= L[-1]; rb = (np.arange(B) + 0.5) / B
for K in (3, 4):
    cuts = [0.0] + [float(np.interp(i / K, L, rb)) for i in range(1, K)] + [1.0 + 1e-9]
    write('set_win_arc', K, [np.where((rank >= cuts[i]) & (rank < cuts[i + 1]))[0] for i in range(K)],
          [dict(cut_lo=round(cuts[i], 4), cut_hi=round(min(cuts[i + 1], 1.0), 4)) for i in range(K)], dict(kind='equal arc length partition', bins=B, smooth_sd_bins=3))

# 4 Gaussian soft windows (stochastic subsample)
sds = [0.05, 0.075, 0.10, 0.13]; rng = np.random.default_rng(0); G = []
for sd in sds:
    G.append(np.where(rng.random(N) < np.exp(-0.5 * ((rank - CENTRE) / sd) ** 2))[0])
write('set_win_kern', len(sds), G, [dict(sd_rank=sd) for sd in sds], dict(kind='Gaussian soft window', centre_rank=CENTRE, seed=0))
