#!/usr/bin/env bash
###############################################################################
# cryoDRGN hyperparameter SWEEP driver  (hudson / GPU)
#
# Runs a small GRID (or random subset) over the two high-impact knobs -- beta
# (KL weight) and zdim -- at D=128, with N seeds each so the scorer can measure
# REPRODUCIBILITY (a state that only appears at one seed is not real).  Each
# trial reuses j1442_recover_states.sh in train-only mode (SKIP_ANALYZE=1,
# SKIP_EXPORT=1), so all it produces is the latent z.<EP>.pkl -- fast.
#
# This does NOT decide anything.  It just produces the runs.  Rank them AFTER
# with:  python scripts/cryodrgn/cryodrgn_sweep_score.py --runs <SWEEP_DIR>/*
#
# WHY only beta & zdim: those control whether subtle states collapse or spread.
# Network topology (NAS) is not the bottleneck.  D is fixed at 128 (your own
# runs: D=128 resolves states, D=256 smooths them).
#
# USAGE (repo root on hudson, cryodrgn env):
#   export IMAGES_DIR=/path/to/parent/of/blob/.mrc      # only needed once (downsample)
#   bash scripts/cryodrgn/cryodrgn_grid_sweep.sh
#
# Trim / expand the grid or dry-run:
#   BETAS="0.01 0.03" ZDIMS="16 24" SEEDS="0 1" DRY_RUN=1 \
#     bash scripts/cryodrgn/cryodrgn_grid_sweep.sh
###############################################################################
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RECOVER="$REPO/scripts/cryodrgn/j1442_recover_states.sh"
J1442_DIR="${J1442_DIR:-$REPO/results_cryodrgn/J1442_real}"

# Grid (override via env).  Default = 3 betas x 2 zdims x 2 seeds = 12 trials.
BETAS="${BETAS:-0.01 0.03 0.06}"
ZDIMS="${ZDIMS:-16 24}"
SEEDS="${SEEDS:-0 1}"
D="${D:-128}"
EPOCHS="${EPOCHS:-50}"
DRY_RUN="${DRY_RUN:-0}"

echo "=============================================================="
echo " cryoDRGN sweep:  betas=[$BETAS]  zdims=[$ZDIMS]  seeds=[$SEEDS]"
echo "   D=$D  epochs=$EPOCHS  (train-only; z.pkl per trial)"
n=0; for b in $BETAS; do for z in $ZDIMS; do for s in $SEEDS; do n=$((n+1)); done; done; done
echo "   total trials: $n"
echo "   output under: $J1442_DIR/train_sweep_*"
echo "=============================================================="

i=0
for BETA in $BETAS; do
  for ZDIM in $ZDIMS; do
    for SEED in $SEEDS; do
      i=$((i+1))
      TAG="train_sweep_D${D}_z${ZDIM}_b${BETA//./p}_s${SEED}"
      OUT="$J1442_DIR/$TAG"
      # resumable: skip trials that already produced the final latent
      if ls "$OUT"/z."$EPOCHS".pkl >/dev/null 2>&1; then
        echo "[$i/$n] $TAG -- already done, skipping"
        continue
      fi
      echo "[$i/$n] $TAG"
      if [[ "$DRY_RUN" == "1" ]]; then
        echo "    DRY_RUN: BETA=$BETA ZDIM=$ZDIM SEED=$SEED D=$D EPOCHS=$EPOCHS TAG=$TAG"
        continue
      fi
      BETA="$BETA" ZDIM="$ZDIM" SEED="$SEED" D="$D" EPOCHS="$EPOCHS" TAG="$TAG" \
        SKIP_ANALYZE=1 SKIP_EXPORT=1 \
        bash "$RECOVER"
    done
  done
done

echo
echo "SWEEP COMPLETE.  Rank the trials (locally or on hudson, cryodrgn env):"
echo "  python scripts/cryodrgn/cryodrgn_sweep_score.py \\"
echo "    --runs $J1442_DIR/train_sweep_* \\"
echo "    -o $J1442_DIR/sweep_leaderboard"
