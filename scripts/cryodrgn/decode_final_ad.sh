#!/usr/bin/env bash
# =============================================================================
# decode_final_ad.sh  --  Options (a)-(d): decode maps from the FINISHED
#                         high-resolution (D=256) cryoDRGN final on the GPU box.
#
# WHEN TO RUN: only AFTER the D=256 final training finishes and has written
# z.50.pkl + weights.50.pkl (e.g. results_cryodrgn/J264_real/train_final_retry/).
# Everything here uses cryoDRGN's DECODER (generative): give it a latent point,
# it paints a 3D map. So none of this is a CryoSPARC re-refinement -- there is no
# circularity. Run on hudson (GPU) because eval_vol at box 256 is far faster there.
#
# WHY EACH OPTION (the reasoning, mapped to the single-basin J264 result):
#   (a) decode ONE map per CryoSPARC class (class medoid in latent space) and
#       cross-correlate all of them  -> "is each CryoSPARC class even a distinct
#       map?"  The pilot already showed the 9 classes collapse to 2 structural
#       groups; this re-tests it at high resolution.
#   (b) k-means "on-data" cluster-center maps (the paper's density-generation
#       step, Zhong 2021) -> an unbiased set of representative maps spanning the
#       whole latent cloud, not tied to CryoSPARC's labels.
#   (c) PC1/PC2 volume traversal -> the HONEST view of a continuous barrier-free
#       axis: a smooth movie morphing from one structural pole to the other.
#   (d) LDA-decoded class endpoints -> the "sophisticated" test: fit a SUPERVISED
#       axis that maximally separates a chosen class pair, decode both extremes,
#       and verify the difference is reproducible on independent data halves.
#
# NOTE ON --ind (IMPORTANT): the final was trained with --ind ind_keep.pkl, so
# z.50 has only the KEPT particles (~220k of 301k) in filtered order. Options (a)
# and (d) map latent rows to CryoSPARC class labels by uid THROUGH the passthrough
# and assume len(passthrough)==len(z). We therefore build an ind-filtered
# passthrough (STEP 0) so the row counts and order match. Options (b)/(c) operate
# on z.pkl directly and need no passthrough, so they are unaffected by --ind.
#
# TO ADAPT for J1442 (train_final) or the fullset runs: change DSET/TRAINDIR/
# CS/PROTEIN_IDX/IND below. For the fullset (no --ind) SKIP STEP 0 and set
# PASS_KEPT="$PASS_FULL" (row counts already match z).
# =============================================================================
set -euo pipefail

cd /home/mae2183/cryoem-classification
PY=/home/mae2183/miniconda3/envs/cryodrgn/bin/python
CD=/home/mae2183/miniconda3/envs/cryodrgn/bin/cryodrgn

# ---- CONFIG (J264 D=256 purified final) -------------------------------------
DSET=J264
TRAINDIR=results_cryodrgn/J264_real/train_final_retry
EPOCH=50                                   # final epoch -> z.$EPOCH.pkl / weights.$EPOCH.pkl
Z=$TRAINDIR/z.$EPOCH.pkl
WEIGHTS=$TRAINDIR/weights.$EPOCH.pkl
CONFIG=$TRAINDIR/config.yaml
PASS_FULL=data/J264/cryosparc_P7_J264_passthrough_particles_all_classes_blob.cs
CS=data/J264/cryosparc_P7_J264_00062_particles_alignments3D_multi.cs
CLASS_DIR=data/J264_classes
IND=results_cryodrgn/J264_real/inputs/ind_keep.pkl   # set IND="" for a fullset run
PROTEIN_IDX="6,7,8,9,10,11,12,13,14"       # 9 protein classes (6 dummy classes)
N_DUMMIES=6
# D=256 downsampled from box 320 @ 0.83 A/pix  ->  apix = 0.83 * 320 / 256
APIX=1.0375
DECODE_BOX=256                             # full res on GPU; drop to 128 for a quick look

PASS_KEPT=$TRAINDIR/passthrough_kept.npy   # produced by STEP 0 (used by a & d)

# ---- STEP 0: build the ind-filtered passthrough (skip if IND is empty) ------
# cryoDRGN's --ind keeps original order, so z row i == passthrough[sorted(ind)][i].
if [ -n "$IND" ]; then
  echo "[step0] building ind-filtered passthrough -> $PASS_KEPT"
  $PY - "$PASS_FULL" "$IND" "$PASS_KEPT" <<'PY'
import sys, pickle, numpy as np
pass_full, ind_path, out = sys.argv[1:4]
pt  = np.load(pass_full)                          # structured .cs array (all particles)
ind = pickle.load(open(ind_path, "rb"))
ind = np.sort(np.asarray(ind).ravel().astype(np.int64))
kept = pt[ind]
np.save(out, kept)                                # np.save writes <out> (already .npy)
print(f"[step0] {len(pt)} -> {len(kept)} kept rows written to {out}")
PY
else
  echo "[step0] IND empty (fullset) -> using full passthrough"
  PASS_KEPT="$PASS_FULL"
fi

# ---- OPTION (a): one decoded map per CryoSPARC class + CC matrix -------------
# medoid = the most central real particle of each class in latent space (robust
# to outliers). --traj 6 also decodes 6 frames along PC1 for context.
echo "[a] per-class decoded maps + cross-correlation matrix"
$PY scripts/cryodrgn/cryodrgn_decode_states.py \
    --dataset "$DSET:$Z:$PASS_KEPT:$CS:$PROTEIN_IDX:$WEIGHTS:$CONFIG:$CLASS_DIR" \
    --n-dummies $N_DUMMIES --rep medoid --traj 6 \
    -d $DECODE_BOX --apix $APIX --run \
    -o results_cryodrgn/decode_states_final

# ---- OPTIONS (b) k-means maps + (c) PC1/PC2/PC3 traversal --------------------
# Both come out of one analyze call: kmeans20/ = option (b) on-data maps;
# pc1/ pc2/ pc3/ = option (c) traversal volumes + plots. Needs no passthrough/cs.
# John asked for PC3 as well (he expects ~3 principal conformational coordinates:
# NBD rocking, exit-portal unfolding, domain dissociation), hence --pc 3.
echo "[b+c] k-means on-data maps + PC1/PC2/PC3 traversal volumes"
$CD analyze "$TRAINDIR" $EPOCH --ksample 20 --pc 3

# Continuous decoding along the ON-MANIFOLD PC traversals from
# cryodrgn_conformational_landscape.py (percentile-window means; smoother than
# analyze's linear PC walk). Optional:
# for PC in 1 2 3; do
#   $CD eval_vol "$WEIGHTS" -c "$CONFIG" --Apix $APIX -d $DECODE_BOX \
#       --zfile results_cryodrgn/conformational_landscape/J264/traversal_zfiles/pc${PC}.txt \
#       --prefix confpc${PC}_vol -o results_cryodrgn/conformational_landscape/J264/vols_pc${PC} \
# done

# ---- OPTION (d): LDA-decoded class endpoints (supervised, split-half checked)-
# Pairs test the two structural groups the pilot found:
#   cross-group (expect a real difference): 6-8, 10-12, 14-9
#   within-group near-duplicate controls (expect ~no difference): 11-14, 12-13
echo "[d] LDA-decoded endpoints for selected class pairs"
$PY scripts/cryodrgn/cryodrgn_lda_states.py \
    --dataset "$DSET:$Z:$PASS_KEPT:$CS:$PROTEIN_IDX:$WEIGHTS:$CONFIG:$CLASS_DIR" \
    --n-dummies $N_DUMMIES --pairs 6-8,10-12,14-9,11-14,12-13 \
    --n-traj 6 -d $DECODE_BOX --apix $APIX --run \
    -o results_cryodrgn/lda_states_final

echo "[done] a-d decoding complete. Sync results_cryodrgn/{decode_states_final,"
echo "       lda_states_final} and $TRAINDIR/analyze.$EPOCH back to the workspace."
