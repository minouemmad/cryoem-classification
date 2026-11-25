#!/bin/bash
# Per-dataset envelopes: consensus envelope (envelope.py) + both CFTR models fitted into the consensus map, both hands (fit_pdb.py)
# -> regions.npz, env64_pdb.npy in $WORK/env/DS.   Usage: envelope.sh GPU DS STACK POSES CTF
source "$(dirname "$(readlink -f "$0")")/../../config/env.sh"; H=$ROOT/pipeline/1_envelope
GPU=$1; DS=$2; O=$WORK/env/$DS; mkdir -p $O; cd $O
[ -f env64.npy ] || CUDA_VISIBLE_DEVICES=$GPU $PY $H/envelope.py $3 $4 $5 $O > env.log 2>&1 || { echo "FAILED env $DS"; exit 1; }
$PY -c "
import numpy as np, mrcfile
m = np.load('consensus64.npy'); ap = float(mrcfile.open('mask64.mrc').voxel_size.x); open('apix.txt', 'w').write('%.5f' % ap)
for nm, a in (('consensus', m), ('consensusflip', m[::-1].copy())):
    with mrcfile.new(nm + '.mrc', overwrite=True) as o: o.set_data(a.astype(np.float32)); o.voxel_size = ap"
timeout 3000 bash -c "$(declare -f cx); cx $H/fit_pdb.py" 2>&1 | grep -E '^MODEL' > fit_out.txt
$PY $H/regions.py > regions.log 2>&1; echo "FIT $DS $(tail -1 regions.log)"
