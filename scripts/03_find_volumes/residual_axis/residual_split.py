"""Residual-axis split of one RAE/VAE model's middle PC1 window, label-free (the recipe of the J4001 / J4624 stress tests)."""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
DS, TAG, SCF, ZF = sys.argv[1:5]; R = Path('/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn')
if DS == 'J4001':
    W = R / 'J4001_gP25_WT'; A = W / 'candidate_groups_epoch50/axis_survey'; base = np.load(A / 'pc_scan_J4001_rae_nuposes.scores.npy')[:, 0]; lab = np.load(W / 'method_rank/j2923_labels.npy')
    comp = lambda g: 'P8 %.2f P7 %.2f P6 %.2f Alt %.2f' % tuple((lab[g] == c).mean() for c in (2, 1, 0, 3))
    ref = [np.load(A / 'set_win_resid/groups/leaves_K2_g0.npy'), np.load(A / 'set_win_resid/groups/leaves_K2_g1.npy')]; OUT = A / f'set_rv_{TAG}'
else:
    J = R / 'J4624_eP37_WT_T2a/recon_tree'; base = np.load(J / 'pc_scan_J4624_rae_nuposes.scores.npy')[:, 0]; lab = np.load(J / 'heteroref_J4624_classes/leaves.npy')
    comp = lambda g: ' '.join('C%02d %.2f' % (6 + c, (lab[g] == c).mean()) for c in range(5)); ref = [np.flatnonzero(np.load(J / 'rs_setA/leaves.npy') == k) for k in (2, 3)]; OUT = J / f'rs_setV{TAG}'
S = np.load(SCF)[:, 0]; N = len(S); sign = float(np.sign(np.corrcoef(S, base)[0, 1])); rank = np.argsort(np.argsort(sign * S)) / N
z = np.load(ZF); assert len(z) == N, (len(z), N); zs = (z - z.mean(0)) / z.std(0)
mid = np.where((rank >= 0.30) & (rank < 0.70))[0]; c = GaussianMixture(2, covariance_type='full', random_state=0, n_init=3).fit(zs[mid][:, :4]).predict(zs[mid][:, :4]); a, b = mid[c == 0], mid[c == 1]
if rank[a].mean() > rank[b].mean(): a, b = b, a
sep = float(np.linalg.norm(zs[a][:, :4].mean(0) - zs[b][:, :4].mean(0))); print(f'{DS} {TAG}: PC1 sign vs base {sign:+.0f} (cc {np.corrcoef(S, base)[0, 1]:+.2f}); mid n {len(mid)}; halves n {len(a)} / {len(b)}; separation {sep:.2f} sd; mean rank {rank[a].mean():.3f} / {rank[b].mean():.3f}')
print('  EVAL g0:', comp(a)); print('  EVAL g1:', comp(b)); print('  EVAL mid:', comp(mid))
ov = [[len(np.intersect1d(x, y)) / len(x) for y in ref] for x in (a, b)]; print('  share of each half in the original-model halves (rows g0,g1; cols orig0,orig1):', np.round(ov, 2).tolist())
OUT.mkdir(parents=True, exist_ok=True)
if DS == 'J4001':
    (OUT / 'groups').mkdir(exist_ok=True); np.save(OUT / 'groups/leaves_K2_g0.npy', np.sort(a)); np.save(OUT / 'groups/leaves_K2_g1.npy', np.sort(b))
else:
    lv = -np.ones(N, np.int16); lv[a] = 0; lv[b] = 1; np.save(OUT / 'leaves.npy', lv)
json.dump(dict(ds=DS, tag=TAG, scores=SCF, z=ZF, n=[int(len(a)), int(len(b))], sep_sd=sep, eval=dict(g0=comp(a), g1=comp(b), mid=comp(mid)), overlap_with_original=ov), open(OUT / 'rv.json', 'w'), indent=1); print('wrote', OUT)
