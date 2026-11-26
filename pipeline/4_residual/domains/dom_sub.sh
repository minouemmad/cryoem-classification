#!/bin/bash
# (11-26) Conditional per-domain residual fits (label-free): inside each window of the J4001 start_free S0 axis (both ends g0 / g8 and the stage-1 pick g4),
# fit the residual model with the eigen-volumes confined to one domain (masks of dom_fit.sh), against the window's OWN mean map.  K=4, 30 iterations, 50-7 A.
# Out: pipe/J4001/dom_sub/<win>_<DOM>/ppca_sub*.  Usage: dom_sub.sh GPU
source "$(dirname "$(readlink -f "$0")")/../../../config/env.sh"; B=$WORK
O=$B/pipe/J4001/dom_sub; mkdir -p $O; read -r _ STK POS CTF _ <<< "$(grep -E '^J4001 ' $DATASETS)"
for W in g8 g0 g4; do
  [ -f $O/idx_$W.npy ] || $PY -c "import numpy as np; m=np.load('$B/od/J4001/win_free_S0_K9/members.npz')['$W']; np.save('$O/idx_$W.npy', np.sort(m))"
  for DOM in NBD1 TMD1 TMD2 NBD2; do
    D=$O/${W}_$DOM; mkdir -p $D; [ -f $D/ppca_sub_z.npy ] && continue; cp $B/pipe/J4001/domain_fit/$DOM/env.npy $D/env.npy
    CUDA_VISIBLE_DEVICES=$1 MCR_SUBSET=$O/idx_$W.npy MCR_STACK=$STK MCR_POSES=$POS MCR_CTF=$CTF \
      MCR_SCORES=$B/pipe/J4001/start_free/start/rand.scores.npy nice $PY $ROOT/pipeline/4_residual/residual_ppca.py $D sub 4 30 7.0 50.0 > $D/run.log 2>&1 || echo "FAILED $W $DOM"
    echo "$W $DOM $(grep -h 'fold-vs-fold subspace cosines' $D/ppca_sub.txt | tail -1)"
  done
done
echo DOMSUB_DONE
