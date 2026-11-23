"""Rewrite group STAR files to point at the local 128 px stack (cryoDRGN order = STAR row order).
Usage: star_local_stack.py [SETS]"""
import sys
from pathlib import Path
import numpy as np, mrcfile
J = Path('/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn/J4624_eP37_WT_T2a'); stack = J / 'inputs/particles.128.mrcs'
with mrcfile.mmap(stack, permissive=True) as m: n, D = m.data.shape[0], m.data.shape[-1]
full = (J / 'nu_consensus/J4624_eP37_WT_T2a_particles.star').read_text().split('\n'); he = next(i for i, l in enumerate(full) if l.startswith('_rlnOpticsGroup #7')) + 1; rows = [l for l in full[he:] if l.strip()]
assert len(rows) == n, (len(rows), n); hdr = []
for l in full[:he]:
    p = l.split()
    if len(p) == 8 and p[7].startswith('opticsGroup'): p[1] = '%.6f' % (float(p[1]) * int(p[5]) / D); p[5] = str(D); l = ' '.join(p)
    hdr.append(l)
new = [' '.join(['%06d@%s' % (i + 1, stack)] + r.split()[1:]) for i, r in enumerate(rows)]; print('stack', stack, n, 'images of', D, 'px; optics lines:', [l for l in hdr if 'opticsGroup' in l and len(l.split()) == 8])
for s in (sys.argv[1].split(',') if len(sys.argv) > 1 else list('ABC')):
    (J / f'recon_tree/rs_set{s}/cryosparc').mkdir(exist_ok=True)
    lv = np.load(J / f'recon_tree/rs_set{s}/leaves.npy')
    for k in range(lv.max() + 1):
        out = J / f'recon_tree/rs_set{s}/cryosparc/eP37_rs{s}d128_leaf{k}.star'; out.write_text('\n'.join(hdr + [r for r, l in zip(new, lv) if l == k]) + '\n\n'); print(out.name, int((lv == k).sum()))
