"""2026-11-19 stress-test windows (user: the recipe is not ready to freeze; the window size is arbitrary; test reproducibility, variability, interpretability)."""
import hashlib, json
from pathlib import Path
import numpy as np
A = Path('/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn/J4001_gP25_WT/candidate_groups_epoch50/axis_survey')
SF = A / 'pc_scan_J4001_rae_nuposes.scores.npy'
S = np.load(SF); N = len(S); s = S[:, 0]
order = np.argsort(s, kind='stable'); rank = np.empty(N); rank[order] = np.arange(N) / N
meta = dict(scores_file='candidate_groups_epoch50/axis_survey/' + SF.name, scores_md5=hashlib.md5(open(SF, 'rb').read()).hexdigest(), N=N)
q6 = (S[:, 5] - S[:, 5].mean()) / S[:, 5].std()


def sd_for(c, n):
    lo, hi = 0.003, 3.0
    for _ in range(70):
        mid = 0.5 * (lo + hi); w = np.exp(-0.5 * ((rank - c) / mid) ** 2)
        lo, hi = (mid, hi) if w.sum() < n else (lo, mid)
    return 0.5 * (lo + hi)


def gauss_window(c, n, rng):
    sd = sd_for(c, n); w = np.exp(-0.5 * ((rank - c) / sd) ** 2); return np.where(rng.random(N) < w)[0], sd


def knn_window(c, n):
    m = (rank >= c - 0.01) & (rank <= c + 0.01); cv = S[m].mean(0); d = np.linalg.norm(S - cv, axis=1); return np.sort(np.argsort(d)[:n]), float(np.sort(d)[n])


def write(name, specs, kind):
    d = A / name / 'groups'; d.mkdir(parents=True, exist_ok=True); K = len(specs); wins = []
    for j, sp in enumerate(specs):
        g, sd = sp['make']()
        np.save(d / f'leaves_K{K}_g{j}.npy', np.sort(g))
        wins.append(dict(j=j, K=K, n=int(len(g)), rank_lo=round(float(rank[g].min()), 4), rank_hi=round(float(rank[g].max()), 4), rank_mean=round(float(rank[g].mean()), 4),
                         pc1_lo=round(float(s[g].min()), 4), pc1_hi=round(float(s[g].max()), 4), sd_rank=round(float(sd), 4), q6_mean=round(float(q6[g].mean()), 3),
                         **{k: v for k, v in sp.items() if k != 'make'}))
        print(name, f'g{j}', {k: wins[-1][k] for k in ('centre', 'n_target', 'n', 'rank_lo', 'rank_hi', 'label') if k in wins[-1]}, flush=True)
    json.dump(dict(meta, kind=kind, windows=wins), open(A / name / 'windows.json', 'w'), indent=1)


rg = np.random.default_rng(6)
spec = []
for c, ns in ((0.40, (22000, 33000, 75000)), (0.445, (22000, 33000, 43000, 75000))):
    for n in ns:
        spec.append(dict(centre=c, n_target=n, label='grid', make=(lambda c=c, n=n: gauss_window(c, n, rg))))
write('set_win_kgrid', spec, 'Gaussian windows, completion of the (centre, size) grid')
write('set_win_kknn', [dict(centre=c, n_target=n, label='knn10d', make=(lambda c=c, n=n: knn_window(c, n))) for c in (0.40, 0.445) for n in (43000, 58000)], '10-D volume-PC nearest-neighbour windows')
spec = []
for n in (43000, 58000):
    for k in range(4):
        spec.append(dict(centre=0.445, n_target=n, label=f'rep{k + 1}', make=(lambda n=n, k=k: gauss_window(0.445, n, np.random.default_rng(7000 + 10 * k + (n // 1000))))))
write('set_win_krep', spec, 'independent draws of the same nominal Gaussian window (resampling replicates)')
