"""Export axis-survey groups (groups/<tag>_K<K>_g<j>.npy) as RELION stars for cryoSPARC import, with the J4001 consensus .cs (blob + ctf + alignments; ab initio + NU re-estimates poses)."""
import json, subprocess, sys
from pathlib import Path
import numpy as np
man_path, ds, prefix = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
N = Path('/mnt/nfs_HuntLabSynologyDS1621CryoEM/hCFTR/Movies/24sep20b/24sep20b/rawdata/CS-24sep20b-wt-control-white-from-january/gmm_exports/recon_tree_v3')
cs = np.load('/data/hCFTR/mae2183/cryoem-classification/data/J4001_gP25_WT/homoref_J4001/cryosparc_P25_J4001_ctf_particles.cs')
out = man_path.parent / 'export'; out.mkdir(exist_ok=True)
man = json.load(open(man_path)) if man_path.exists() else []
done = {m['name'] for m in man}
for spec in sys.argv[4:]:
    tag, K = spec.split(':'); K = int(K)
    for j in range(K):
        name = f"STD_{prefix}{tag.replace('_quantile', 'q').replace('_gmm', 'g')}K{K}_g{j}"
        if name in done:
            continue
        take = np.load(ds / 'groups' / f'{tag}_K{K}_g{j}.npy')
        with open(out / f'{name}.cs', 'wb') as fh: np.save(fh, cs[take])
        subprocess.run(['/home/mae2183/miniconda3/envs/cryodrgn/bin/python', str(Path(__file__).resolve().parents[1] / '00_dataset_setup' / 'cs2star.py'),
                        str(out / f'{name}.cs'), str(N / f'{name}.star')], check=True, capture_output=True)
        man.append(dict(name=name, group=f'{ds.name}/{tag}_K{K}_g{j}', n=int(len(take)), poses='J4001'))
json.dump(man, open(man_path, 'w'), indent=1)
print(len(man), 'entries in', man_path)
