#!/usr/bin/env bash
###############################################################################
# J2708 (eP30, D1212 ATP/VX) "high-recall" state-recovery run w/ AUTO-RESUME
# Mirrors j1442_recover_states.sh / j264_recover_states.sh for this dataset.
# Automatically finds latest VALID weights and resumes from checkpoint.
###############################################################################
set -euo pipefail

# --------------------------------------------------------------------------- #
# Config (override via environment variables)
# --------------------------------------------------------------------------- #
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYCLUSTER="${PYCLUSTER:-python}"
CRYODRGN="${CRYODRGN:-cryodrgn}"

# Base dir for J2708 cryoDRGN artifacts (hudson convention: J2708_real)
J2708_DIR="${J2708_DIR:-$REPO/results_cryodrgn/J2708_real}"

# Inputs (particles.128.mrcs already exists -> downsample is skipped)
PASS="${PASS:-$REPO/data/eP30W12_J2708/cryosparc_P30_J2708_passthrough_particles_all_classes_blob.cs}"
POSES="${POSES:-$J2708_DIR/inputs/poses.pkl}"
CTF="${CTF:-$J2708_DIR/inputs/ctf.pkl}"
IMAGES_DIR="${IMAGES_DIR:-/home/mae2183/cryoem-classification}"   # only used if downsampling

# Hyperparameters (same "sweet spot" found for J1442/J264: zdim16 beta0.03 D128)
D="${D:-128}"
ZDIM="${ZDIM:-16}"
BETA="${BETA:-0.03}"
EPOCHS="${EPOCHS:-50}"
ENC_DIM="${ENC_DIM:-1024}"
ENC_LAYERS="${ENC_LAYERS:-3}"
DEC_DIM="${DEC_DIM:-512}"
DEC_LAYERS="${DEC_LAYERS:-3}"
BATCH="${BATCH:-64}"
NWORKERS="${NWORKERS:-4}"
SEED="${SEED:-0}"
IND="${IND:-}"
KCLUST="${KCLUST:-10}"

# subset (divide-and-conquer): retrain on just one blob's particles
if [[ -n "$IND" ]]; then
  [[ -f "$IND" ]] || { echo "IND file not found: $IND" >&2; exit 1; }
  IND_TRAIN="--ind $IND"
  IND_KEEP="--ind-keep $IND"
  SUBSUFFIX="_$(basename "${IND%.pkl}")"
else
  IND_TRAIN=""; IND_KEEP=""; SUBSUFFIX=""
fi

# Output layout
TAG="${TAG:-train_recover_D${D}_z${ZDIM}_b${BETA//./p}${SUBSUFFIX}}"
OUT="$J2708_DIR/$TAG"
PARTICLES="${PARTICLES:-$J2708_DIR/inputs/particles.${D}.mrcs}"
CLUSTERS="$OUT/cluster_exports_k${KCLUST}"
mkdir -p "$OUT"

echo "=============================================================="
echo " J2708 high-recall run with AUTO-RESUME & VALIDATION"
echo "   D=$D  zdim=$ZDIM  beta=$BETA  epochs=$EPOCHS"
echo "   enc=${ENC_DIM}x${ENC_LAYERS}  dec=${DEC_DIM}x${DEC_LAYERS}  batch=$BATCH"
echo "   out=$OUT"
echo "=============================================================="

# --------------------------------------------------------------------------- #
# 0) Sanity checks (PASS/IMAGES_DIR only needed if we must downsample/export)
# --------------------------------------------------------------------------- #
for f in "$POSES" "$CTF"; do
  [[ -f "$f" ]] || { echo "MISSING input: $f" >&2; exit 1; }
done
if [[ ! -f "$PARTICLES" ]]; then
  [[ -f "$PASS" ]]       || { echo "MISSING $PASS (needed to downsample)" >&2; exit 1; }
  [[ -d "$IMAGES_DIR" ]] || { echo "IMAGES_DIR not a directory: $IMAGES_DIR" >&2; exit 1; }
fi

# --------------------------------------------------------------------------- #
# 1) Downsample particle stack to D (skip if present)
# --------------------------------------------------------------------------- #
if [[ -f "$PARTICLES" ]]; then
  echo "[1/4] downsample: reuse existing $PARTICLES"
else
  echo "[1/4] downsample -> $PARTICLES"
  "$CRYODRGN" downsample "$PASS" -D "$D" --datadir "$IMAGES_DIR" -o "$PARTICLES"
fi

# --------------------------------------------------------------------------- #
# 2) Find the latest VALID checkpoint
# --------------------------------------------------------------------------- #
echo "[2/4] Looking for valid checkpoint in $OUT"

LOAD_FLAG=""
LATEST_EPOCH=""
VALID_CHECKPOINT=""

if [[ -d "$OUT" ]]; then
  WEIGHTS_LIST=$(ls -1 "$OUT"/weights.*.pkl 2>/dev/null || true)

  if [[ -n "$WEIGHTS_LIST" ]]; then
    WEIGHTS_FILES=$(echo "$WEIGHTS_LIST" | sed -E 's/.*weights\.([0-9]+)\.pkl/\1 &/' | sort -rn | awk '{print $2}')

    if [[ -n "$WEIGHTS_FILES" ]]; then
      echo "  Found $(echo "$WEIGHTS_FILES" | wc -l) checkpoint files"

      for wfile in $WEIGHTS_FILES; do
        WFILENAME=$(basename "$wfile")
        WEPOCH=$(echo "$WFILENAME" | sed -E 's/weights\.([0-9]+)\.pkl/\1/')

        echo "  Testing checkpoint epoch $WEPOCH: $WFILENAME"

        if python -c "
import torch
import sys
try:
    ckpt = torch.load('$wfile', map_location='cpu', weights_only=False)
    if 'model_state_dict' in ckpt or 'epoch' in ckpt:
        sys.exit(0)
    else:
        sys.exit(1)
except Exception as e:
    sys.exit(1)
" 2>/dev/null; then
          echo "  Valid checkpoint found: epoch $WEPOCH"
          VALID_CHECKPOINT="$wfile"
          LATEST_EPOCH="$WEPOCH"
          break
        else
          echo "  Checkpoint epoch $WEPOCH is corrupted or invalid"
        fi
      done
    else
      echo "  No checkpoint files found. Starting from scratch."
    fi
  else
    echo "  No checkpoint files found. Starting from scratch."
  fi
else
  echo "  No checkpoint files found. Starting from scratch."
fi

if [[ -n "$VALID_CHECKPOINT" ]]; then
  if [[ "$LATEST_EPOCH" -ge "$EPOCHS" ]]; then
    echo "  Already completed $EPOCHS epochs. Skipping training."
    if [[ -f "$OUT/z.${EPOCHS}.pkl" ]]; then
      ZFINAL="$OUT/z.${EPOCHS}.pkl"
      echo "      final latent: $ZFINAL"
    fi
  else
    echo "  Resuming from epoch $LATEST_EPOCH (weights: $(basename "$VALID_CHECKPOINT"))"
    LOAD_FLAG="--load $VALID_CHECKPOINT"
  fi
else
  echo "  No valid checkpoints found. Starting from scratch."
fi

# --------------------------------------------------------------------------- #
# 3) train_vae (skip if already at target epochs)
# --------------------------------------------------------------------------- #
if [[ -n "$VALID_CHECKPOINT" ]] && [[ "$LATEST_EPOCH" -ge "$EPOCHS" ]] && [[ -f "$OUT/z.${EPOCHS}.pkl" ]]; then
  echo "[3/4] Training already complete. Skipping."
else
  echo "[3/4] train_vae -> $OUT"

  TRAIN_CMD="$CRYODRGN train_vae $PARTICLES \
    --ctf $CTF --poses $POSES \
    --zdim $ZDIM --beta $BETA --num-epochs $EPOCHS --seed $SEED \
    $IND_TRAIN \
    --enc-dim $ENC_DIM --enc-layers $ENC_LAYERS \
    --dec-dim $DEC_DIM --dec-layers $DEC_LAYERS \
    --batch-size $BATCH --lazy --num-workers $NWORKERS --no-amp \
    --checkpoint 1"

  if [[ -n "$LOAD_FLAG" ]]; then
    TRAIN_CMD="$TRAIN_CMD $LOAD_FLAG"
  fi

  TRAIN_CMD="$TRAIN_CMD -o $OUT"

  echo "  Running: $TRAIN_CMD"
  eval "$TRAIN_CMD"
fi

# --------------------------------------------------------------------------- #
# 4) Find the final z file
# --------------------------------------------------------------------------- #
if [[ -f "$OUT/z.${EPOCHS}.pkl" ]]; then
  ZFINAL="$OUT/z.${EPOCHS}.pkl"
  echo "      final latent: $ZFINAL"
else
  ZFINAL="$(ls -1 "$OUT"/z.*.pkl 2>/dev/null | sed -E 's/.*z\.([0-9]+)\.pkl/\1 &/' | sort -n | tail -1 | awk '{print $2}')"
  if [[ -z "${ZFINAL:-}" ]]; then
    echo "ERROR: no z.*.pkl produced" >&2
    exit 1
  fi
  echo "      latest latent: $ZFINAL (not target epoch $EPOCHS)"
fi

# --------------------------------------------------------------------------- #
# 5) analyze (optional; needed for kmeans20 volumes used by make_focus_mask.py)
# --------------------------------------------------------------------------- #
EP="$(basename "$ZFINAL" | sed -E 's/z\.([0-9]+)\.pkl/\1/')"
if [[ "${SKIP_ANALYZE:-0}" == "1" ]]; then
  echo "[4/5] analyze: SKIPPED (SKIP_ANALYZE=1)"
else
  echo "[4/5] analyze epoch $EP"
  "$CRYODRGN" analyze "$OUT" "$EP" || echo "  (analyze failed/optional -- continuing)"
fi

# --------------------------------------------------------------------------- #
# 6) Export clusters (optional)
# --------------------------------------------------------------------------- #
if [[ "${SKIP_EXPORT:-0}" == "1" ]]; then
  echo "[5/5] over-cluster export: SKIPPED (SKIP_EXPORT=1)"
  echo "      latent ready: $ZFINAL"
  exit 0
fi

[[ -f "$PASS" ]] || { echo "MISSING $PASS (needed for cluster export)" >&2; exit 1; }
echo "[5/5] over-cluster (full z, k=$KCLUST) -> $CLUSTERS"
"$PYCLUSTER" "$REPO/scripts/cryodrgn/export_latent_clusters.py" \
  --z "$ZFINAL" \
  --passthrough-cs "$PASS" \
  $IND_KEEP \
  -k "$KCLUST" --dataset J2708 \
  --min-resp 0.8 \
  -o "$CLUSTERS"

cat <<EOF

==============================================================
DONE.  Next steps (CryoSPARC):
  1. Import each $CLUSTERS/J2708_cluster_c*.cs  (blob+CTF present).
  2. Ab-initio (K=1) -> NU-refine each cluster.
  3. Merge duplicates: compare the NU maps with
       python scripts/cryodrgn/compare_maps.py ...
     (rigid-align first -- independent refinements sit in their own pose
     frames, ~180 deg offsets are normal).
  4. Report the MERGED, ab-initio-confirmed states.
==============================================================
EOF
