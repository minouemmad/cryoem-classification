#!/usr/bin/env bash
###############################################################################
# J1442-homorefine ZDIM / BETA "concentration" sweep — hudson
#
# WHY: the honest-pose (J4001 consensus) latent captured the REAL motion
# (latent PC1 vs pose-free 3DVA |r|=0.48, canonical r=0.59) but at zdim=16 the
# weak signal DIFFUSES across all 16 dims (participation-dim 15.7/16) -> no
# clusters. Shrinking zdim FORCES that variance into PC1; lowering beta relaxes
# the KL pull. This sweep finds the setting where the real motion concentrates
# enough to resolve. Complements (does NOT duplicate) focused_pipeline_*.sh,
# which only ever trains zdim16/beta0.03.
#
# Trains on the FOCUSED residual stack by default (concentration lever #1 +
# #2 together); falls back to the plain homorefine stack if the residual isn't
# built yet. Reuses j1442_homorefine_recover_states.sh for every train.
#
# USAGE (repo root on hudson, cryodrgn env):
#   export IMAGES_DIR=/path/to/parent/of/J995
#   bash scripts/cryodrgn/j1442_homorefine_zdim_sweep.sh
#
# Override the grid / stack without editing:
#   ZDIMS="2 4 8" BETAS="0.01 0.03" bash scripts/cryodrgn/j1442_homorefine_zdim_sweep.sh
#   PARTICLES=.../inputs/particles.128.mrcs bash ...        # force plain stack
#   SEEDS="0 1" bash ...                                    # 2 seeds => reproducibility score
###############################################################################
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python}"
RECOVER="$REPO/scripts/cryodrgn/j1442_homorefine_recover_states.sh"
SCORER="$REPO/scripts/cryodrgn/cryodrgn_sweep_score.py"

J1442_DIR="${J1442_DIR:-$REPO/results_cryodrgn/J1442_gP25_WT_POSE_BIAS_homorefine}"

# --- pick the training stack: focused residual if present, else plain ---
RESID="$J1442_DIR/inputs/particles.128.focus.mrcs"
PLAIN="$J1442_DIR/inputs/particles.128.mrcs"
if [[ -z "${PARTICLES:-}" ]]; then
  if [[ -f "$RESID" ]]; then
    PARTICLES="$RESID"; STACK="focus"
    echo "[stack] using FOCUSED residual: $RESID"
  else
    PARTICLES="$PLAIN"; STACK="plain"
    echo "[stack] focused residual not found -> using PLAIN stack: $PLAIN"
    echo "        (run focused_pipeline_j1442_homorefine.sh steps 1-2 first to"
    echo "         sweep the focused stack, which is the stronger lever.)"
  fi
else
  case "$PARTICLES" in *focus*) STACK="focus";; *) STACK="plain";; esac
  echo "[stack] using caller-supplied PARTICLES=$PARTICLES"
fi

# --- sweep grid ---
ZDIMS="${ZDIMS:-4 8 16}"
BETAS="${BETAS:-0.01 0.03}"
SEEDS="${SEEDS:-0}"
EPOCHS="${EPOCHS:-50}"
D="${D:-128}"
SEP_THRESH="${SEP_THRESH:-1.5}"

PREFIX="train_${STACK}_sweep"
echo "=============================================================="
echo " J1442-homorefine concentration sweep"
echo "   stack=$STACK  zdims=[$ZDIMS]  betas=[$BETAS]  seeds=[$SEEDS]"
echo "   epochs=$EPOCHS  D=$D  tag prefix=$PREFIX"
echo "=============================================================="

for Z in $ZDIMS; do
  for B in $BETAS; do
    for S in $SEEDS; do
      TAG="${PREFIX}_z${Z}_b${B//./p}_s${S}"
      if [[ -f "$J1442_DIR/$TAG/z.$((EPOCHS-1)).pkl" || -f "$J1442_DIR/$TAG/z.${EPOCHS}.pkl" ]]; then
        echo "== skip (done): $TAG =="
        continue
      fi
      echo "== train: zdim=$Z beta=$B seed=$S -> $TAG =="
      PARTICLES="$PARTICLES" TAG="$TAG" \
        ZDIM="$Z" BETA="$B" D="$D" EPOCHS="$EPOCHS" SEED="$S" SKIP_EXPORT=1 \
        bash "$RECOVER"
    done
  done
done

echo "== score the whole sweep (label-free: resolvable modes + reproducibility) =="
"$PY" "$SCORER" \
  --runs "$J1442_DIR/${PREFIX}_z"* \
  -o "$J1442_DIR/${PREFIX}_leaderboard" --sep-thresh "$SEP_THRESH"

cat <<EOF

DONE.  Read $J1442_DIR/${PREFIX}_leaderboard for the ranking.
Look for a (zdim,beta) where RESOLVABLE MODES >= 2 AND (with SEEDS="0 1")
the reproducibility canonical-corr is high -- that is a REAL split, not a
seed artifact. Confirm the winner's PC1 tracks the pose-free 3DVA axis before
committing GPU to per-cluster CryoSPARC refinement.
EOF
