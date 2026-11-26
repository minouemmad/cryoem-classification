"""(11-26) Split one window subset along the leading component of its conditional domain fit (median split, label-free) and write the STARs
like win_build.py.  Usage: sub_split.py WIN DOM   (J4001, start_free S0 windows; z from pipe/J4001/dom_sub/<WIN>_<DOM>/ppca_sub_z.npy)"""
import json, os, sys
from pathlib import Path
import numpy as np, mrcfile
R = Path(os.environ.get('RES', '/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn')); B = R / 'model_pipeline'; ds = 'J4001'; W = R / 'J4001_gP25_WT'
win, dom = sys.argv[1], sys.argv[2]; D0 = B / f'pipe/J4001/dom_sub'; idx = np.load(D0 / f'idx_{win}.npy'); z = np.load(D0 / f'{win}_{dom}/ppca_sub_z.npy')[:, 0]
assert len(z) == len(idx); med = np.median(z); groups = [idx[z <= med], idx[z > med]]
star = B / 'od/J4001/consensus_J4001.star'; stack = W / 'inputs/particles.128.mrcs'
with mrcfile.mmap(stack, permissive=True) as mm: n, D = mm.data.shape[0], mm.data.shape[-1]
full = star.read_text().split('\n'); dp = next(i for i, l in enumerate(full) if l.strip() == 'data_particles')
he = next(i for i in range(dp + 1, len(full)) if full[i].strip() and not full[i].startswith('_') and full[i].strip() != 'loop_'); rows = [l for l in full[he:] if l.strip()]
assert len(rows) == n; hdr = []
for l in full[:he]:
    p = l.split()
    if len(p) == 8 and p[7].startswith('opticsGroup'): p[1] = '%.6f' % (float(p[1]) * int(p[5]) / D); p[5] = str(D); l = ' '.join(p)
    hdr.append(l)
name = f'dsub_{win}_{dom}'; K = 2; O = B / f'od/{ds}/{name}_K{K}'; (O / 'cryosparc').mkdir(parents=True, exist_ok=True); out = []
for k, g in enumerate(groups):
    f = O / f'cryosparc/{ds}_{name}_K{K}_g{k}.star'; f.write_text('\n'.join(hdr + [' '.join(['%06d@%s' % (i + 1, stack)] + rows[i].split()[1:]) for i in g]) + '\n\n'); out.append(str(f))
np.savez(O / 'members.npz', g0=groups[0], g1=groups[1])
json.dump(dict(ds=ds, run=name, K=K, names=None, sizes=[int(len(g)) for g in groups], stars=out, axis=f'win_free_S0_K9 {win}, {dom} conditional component 1, median split'), open(O / 'set.json', 'w'), indent=1)
print(name, 'K', K, [len(g) for g in groups])
