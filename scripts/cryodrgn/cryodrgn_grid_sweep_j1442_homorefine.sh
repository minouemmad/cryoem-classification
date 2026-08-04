#!/usr/bin/env bash
###############################################################################
# cryoDRGN hyperparameter SWEEP driver for J1442-homorefine (J4001)  (hudson)
#
# J1442_homorefine = fresh homogeneous refinement of the same 230,396 J1442/
# J1497 particles with a single global-consensus alignments3D/pose (an
# independent pose frame vs the original J1442_real run -- both are pose-safe,
# this is a robustness check, not a bias fix). Use this sweep to find the
# reproducible-mode count on THIS pose frame and compare against the original
# J1442 sweep leaderboard (results_cryodrgn/J1442_gP25_WT_POSE_BIAS/train_sweep_*).
#
# Mirrors cryodrgn_grid_sweep_j399.sh but drives
# j1442_homorefine_recover_states.sh. Runs a grid over beta (KL weight) and
# zdim at D=128, N seeds each, so the scorer can gate on REPRODUCIBILITY (a
# mode that only appears at one seed is not real). Each trial is train-only
# (SKIP_ANALYZE=1, SKIP_EXPORT=1) -> just z.<EP>.pkl, fast.
#
# FIRST TRIAL derives poses.pkl/ctf.pkl + downsamples particles.128.mrcs
# (J1442-homorefine has no prior cryoDRGN preprocessing); every later trial
# reuses them.
#
# Rank AFTER with the label-free, reproducibility-gated scorer:
#   python scripts/cryodrgn/cryodrgn_sweep_score.py \
#     --runs results_cryodrgn/J1442_gP25_WT_POSE_BIAS_homorefine/train_sweep_* \
#     -o results_cryodrgn/J1442_gP25_WT_POSE_BIAS_homorefine/sweep_leaderboard
# Pick the config with the MOST REPRODUCIBLE modes (NOT the most UMAP blobs --
# low beta adds noise dims, not states).
#
# USAGE (repo root on hudson, cryodrgn env):
#   export IMAGES_DIR=/path/to/parent/of/J995   # REQUIRED (same images as the original J1442)
#   bash scripts/cryodrgn/cryodrgn_grid_sweep_j1442_homorefine.sh
#   BETAS="0.01 0.03" ZDIMS="16 24" SEEDS="0 1" DRY_RUN=1 \
#     bash scripts/cryodrgn/cryodrgn_grid_sweep_j1442_homorefine.sh
###############################################################################
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RECOVER="$REPO/scripts/cryodrgn/j1442_homorefine_recover_states.sh"
J1442_DIR="${J1442_DIR:-$REPO/results_cryodrgn/J1442_gP25_WT_POSE_BIAS_homorefine}"

# Grid (override via env). Default = 3 betas x 2 zdims x 2 seeds = 12 trials.
BETAS="${BETAS:-0.01 0.03 0.06}"
ZDIMS="${ZDIMS:-16 24}"
SEEDS="${SEEDS:-0 1}"
D="${D:-128}"
EPOCHS="${EPOCHS:-50}"
DRY_RUN="${DRY_RUN:-0}"

echo "=============================================================="
echo " J1442-homorefine cryoDRGN sweep:  betas=[$BETAS]  zdims=[$ZDIMS]  seeds=[$SEEDS]"
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
