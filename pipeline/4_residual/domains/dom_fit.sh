#!/bin/bash
# (11-25) Per-domain residual models for J4001 (label-free): the same residual model as the pipeline's first step (mp_ppca.py, random bins = consensus reference,
# K=4 components, 30 EM iterations, 50-7 A, two-fold), with the eigen-volumes confined to ONE domain: TMD1 / NBD1 / TMD2 / NBD2 from the fitted V-shape model
# (method_rank/knee/regions.npz), each dilated by 2 voxels (~8 A) and kept inside the protein envelope.  Out: pipe/J4001/domain_fit/<DOM>/ppca_dom_*.  Usage: dom_fit.sh GPU
source "$(dirname "$(readlink -f "$0")")/../../../config/env.sh"; B=$WORK; R=$RES
O=$B/pipe/J4001/domain_fit; mkdir -p $O; read -r _ STK POS CTF ENV64 _ <<< "$(grep -E '^J4001 ' $DATASETS)"
for DOM in TMD1 NBD1 TMD2 NBD2; do
  mkdir -p $O/$DOM; [ -f $O/$DOM/ppca_dom_z.npy ] && continue
  $PY -c "
import numpy as np; from scipy.ndimage import binary_dilation
r = np.load('$R/J4001_gP25_WT/method_rank/knee/regions.npz'); e = np.load('$ENV64').astype(bool); m = binary_dilation(r['$DOM'], iterations=2) & e
print('$DOM mask voxels (64^3)', int(m.sum()), 'of envelope', int(e.sum()))
np.save('$O/$DOM/env.npy', np.repeat(np.repeat(np.repeat(m, 2, 0), 2, 1), 2, 2))"
  CUDA_VISIBLE_DEVICES=$1 MCR_STACK=$STK MCR_POSES=$POS MCR_CTF=$CTF \
    MCR_SCORES=$B/pipe/J4001/start_free/start/rand.scores.npy nice $PY $ROOT/pipeline/4_residual/residual_ppca.py $O/$DOM dom 4 30 7.0 50.0 > $O/$DOM/run.log 2>&1 || echo "FAILED $DOM"
  echo "$DOM $(grep -h 'fold-vs-fold subspace cosines' $O/$DOM/ppca_dom.txt | tail -1)"
done
echo DOMFIT_DONE
