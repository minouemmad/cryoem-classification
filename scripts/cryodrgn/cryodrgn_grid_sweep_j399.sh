#!/usr/bin/env bash
###############################################################################
# cryoDRGN hyperparameter SWEEP driver for J399  (hudson / GPU)
#
# J399 = homogeneous refinement of the J264 particle set with a TRUE single
# global-consensus alignments3D/pose (pose-safe, unlike J264's per-class
# poses -- see repo memory "POSE-BIAS FINDING" / "J264 HOMOREFINE"). Use this
# sweep to find the reproducible-mode count on the UNBIASED pose frame and
# compare it against the J264 sweep leaderboard.
#
# Mirrors cryodrgn_grid_sweep_j264.sh but drives j399_recover_states.sh. Runs
# a grid over beta (KL weight) and zdim at D=128, N seeds each, so the scorer
# can gate on REPRODUCIBILITY (a mode that only appears at one seed is not
# real). Each trial is train-only (SKIP_ANALYZE=1, SKIP_EXPORT=1) -> just
# z.<EP>.pkl, fast.
#
# FIRST TRIAL derives poses.pkl/ctf.pkl + downsamples particles.128.mrcs
# (J399 has no prior cryoDRGN preprocessing); every later trial reuses them.
#
# Rank AFTER with the label-free, reproducibility-gated scorer:
#   python scripts/cryodrgn/cryodrgn_sweep_score.py \
#     --runs results_cryodrgn/J399_real/train_sweep_* -o results_cryodrgn/J399_real/sweep_leaderboard
# Pick the config with the MOST REPRODUCIBLE modes (NOT the most UMAP blobs --
# low beta adds noise dims, not states).
#
# USAGE (repo root on hudson, cryodrgn env):
#   export IMAGES_DIR=/path/to/parent/of/J226   # REQUIRED (same images as J264/J235)
#   bash scripts/cryodrgn/cryodrgn_grid_sweep_j399.sh
#   BETAS="0.01 0.03" ZDIMS="16 24" SEEDS="0 1" DRY_RUN=1 \
#     bash scripts/cryodrgn/cryodrgn_grid_sweep_j399.sh
###############################################################################
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RECOVER="$REPO/scripts/cryodrgn/j399_recover_states.sh"
J399_DIR="${J399_DIR:-$REPO/results_cryodrgn/J399_real}"

# Grid (override via env). Default = 3 betas x 2 zdims x 2 seeds = 12 trials.
BETAS="${BETAS:-0.01 0.03 0.06}"
ZDIMS="${ZDIMS:-16 24}"
SEEDS="${SEEDS:-0 1}"
D="${D:-128}"
EPOCHS="${EPOCHS:-50}"
DRY_RUN="${DRY_RUN:-0}"

echo "=============================================================="
echo " J399 cryoDRGN sweep:  betas=[$BETAS]  zdims=[$ZDIMS]  seeds=[$SEEDS]"
echo "   D=$D  epochs=$EPOCHS  (train-only; z.pkl per trial)"
n=0; for b in $BETAS; do for z in $ZDIMS; do for s in $SEEDS; do n=$((n+1)); done; done; done
echo "   total trials: $n"
echo "   output under: $J399_DIR/train_sweep_*"
echo "=============================================================="

i=0
for BETA in $BETAS; do
  for ZDIM in $ZDIMS; do
    for SEED in $SEEDS; do
      i=$((i+1))
      TAG="train_sweep_D${D}_z${ZDIM}_b${BETA//./p}_s${SEED}"
      OUT="$J399_DIR/$TAG"
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
echo "    --runs $J399_DIR/train_sweep_* \\"
echo "    -o $J399_DIR/sweep_leaderboard"
