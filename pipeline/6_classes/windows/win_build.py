"""(11-25) Variable-size soft windows along every axis the pipeline produces (label-free).  Axis j = column j of pipe/<DS>/<RUN>/iter2/S_final.npy
(S0 = the sharpened first axis, r 0.99 with the start; S1, S2 = the next search coordinates).  Particle i is kept with probability exp(-(rank_i - c)^2 / 2 sd^2).
  stage 1 (position): sd 0.10 (~58k, inside the 23k-113k range where P7 quality was flat on 11-16), centres 0.1 .. 0.9   -> run win_<m>_S<j>, K = 9
  stage 2 (size):     at the stage-1 pick c*, sd 0.04 0.06 0.08 0.13 0.17 0.22 (+ the stage-1 sd 0.10 map)            -> run win_<m>_S<j>_c<c*>, K = 6
Writes od/<DS>/<run>_K<K>/{set.json, cryosparc/<DS>_<run>_K<K>_g<k>.star} exactly as od_stars.py does (consensus STAR rows, local 128 px stack).
Usage: win_build.py DS RUN j stage1  |  win_build.py DS RUN j stage2 CENTRE"""
import os
import json, sys
from pathlib import Path
import numpy as np, mrcfile
R = Path(os.environ.get("RES", "/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn")); B = R / 'model_pipeline'
DIR = {'J207': 'J207_hP12_IDOR4', 'J2708': 'J2708_eP30_F508del_E1371W', 'J399': 'J399_hP7_E1371Q_ATP', 'J4624': 'J4624_eP37_WT_T2a', 'J4001': 'J4001_gP25_WT'}
ds, run, j, stage = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]; W = R / DIR[ds]; m = run.replace('start_', '')
S = np.load(B / f'pipe/{ds}/{run}/iter2/S_final.npy')[:, j]; N = len(S); rank = np.argsort(np.argsort(S)) / (N - 1)
if stage == 'stage1': name, spec = f'win_{m}_S{j}', [dict(centre=round(c, 2), sd=0.10) for c in np.arange(0.1, 0.91, 0.1)]
else: c = float(sys.argv[5]); name, spec = f'win_{m}_S{j}_c{int(round(c * 100)):02d}', [dict(centre=c, sd=s) for s in (0.04, 0.06, 0.08, 0.13, 0.17, 0.22)]
star = B / 'od/J4001/consensus_J4001.star' if ds == 'J4001' else next(p for p in [W / f'nu_consensus/{DIR[ds]}_particles.star', W / f'nu_consensus/{ds}_particles.star'] if p.exists())
stack = W / 'inputs/particles.128.mrcs'
with mrcfile.mmap(stack, permissive=True) as mm: n, D = mm.data.shape[0], mm.data.shape[-1]
full = star.read_text().split('\n'); dp = next(i for i, l in enumerate(full) if l.strip() == 'data_particles')
he = next(i for i in range(dp + 1, len(full)) if full[i].strip() and not full[i].startswith('_') and full[i].strip() != 'loop_'); rows = [l for l in full[he:] if l.strip()]
assert len(rows) == n == N, (len(rows), n, N); hdr = []
for l in full[:he]:
    p = l.split()
    if len(p) == 8 and p[7].startswith('opticsGroup'): p[1] = '%.6f' % (float(p[1]) * int(p[5]) / D); p[5] = str(D); l = ' '.join(p)
    hdr.append(l)
new = [' '.join(['%06d@%s' % (i + 1, stack)] + r.split()[1:]) for i, r in enumerate(rows)]
K = len(spec); O = B / f'od/{ds}/{name}_K{K}'; (O / 'cryosparc').mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(1000 + j); out, sizes, idx = [], [], {}
for k, sp in enumerate(spec):
    keep = np.flatnonzero(rng.random(N) < np.exp(-0.5 * ((rank - sp['centre']) / sp['sd']) ** 2)); idx[f'g{k}'] = keep
    f = O / f'cryosparc/{ds}_{name}_K{K}_g{k}.star'; f.write_text('\n'.join(hdr + [new[i] for i in keep]) + '\n\n'); out.append(str(f)); sizes.append(int(len(keep)))
np.savez(O / 'members.npz', **idx)
json.dump(dict(ds=ds, run=name, K=K, names=None, sizes=sizes, stars=out, windows=spec, axis=f'{run} S_final[:, {j}]', kind='Gaussian soft window', seed=1000 + j), open(O / 'set.json', 'w'), indent=1)
print(name, 'K', K, sizes)
