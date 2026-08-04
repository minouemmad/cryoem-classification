#!/usr/bin/env bash
###############################################################################
# J399 "high-recall" state-recovery run with AUTO-RESUME & VALIDATION
#
# J399 = homogeneous refinement of the SAME particle set as J264 (hP7W2, P7
# project), kept for a TRUE global-consensus alignment (single alignments3D/
# pose field, NOT the per-class alignments3D_multi pose that biased the
# original J264 poses -- see repo memory "POSE-BIAS FINDING"). 299,641
# particles, box 320, psize 0.83 A, blob/path -> J226/reconstructed/*.mrc
# (same images as J264/J235).
#
# Unlike j264_recover_states.sh (which assumed poses.pkl/ctf.pkl already
# existed), THIS script also derives them the first time it runs (Step 0),
# since J399 is a brand-new dataset with no prior cryoDRGN preprocessing.
#
# Mirrors j264_recover_states.sh / j1442_recover_states.sh otherwise: same
# tuned "high-recall" hyperparameters (D=128, zdim=16, beta=0.03, over-cluster
# k=10), same STOCK cryodrgn train_vae flags only (no source edits).
#
# USAGE (on hudson, in the cryodrgn env, from the repo root):
#   export IMAGES_DIR=/path/to/parent/of/J226   # parent dir containing J226/reconstructed/*.mrc
#   bash scripts/cryodrgn/j399_recover_states.sh
#
# Override any hyperparameter without editing the file, e.g.:
#   ZDIM=24 BETA=0.06 SEED=1 bash scripts/cryodrgn/j399_recover_states.sh
###############################################################################
set -euo pipefail

# --------------------------------------------------------------------------- #
# Config (override via environment variables)
# --------------------------------------------------------------------------- #
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYCLUSTER="${PYCLUSTER:-python}"
CRYODRGN="${CRYODRGN:-cryodrgn}"

# Base dir for J399 cryoDRGN artifacts
J399_DIR="${J399_DIR:-$REPO/results_cryodrgn/J399_real}"

# MAIN particles file: single-class alignments3D/pose + blob + ctf all in one
# (per repo memory "use MAIN particles file NOT passthrough" for homorefine
# exports -- the _blob.cs passthrough here has NO ctf/pose fields).
MAIN="${MAIN:-$REPO/data/hP7W2_J399/cryosparc_P7_J399_particles_ctf.cs}"
PASS="${PASS:-$MAIN}"                 # same file works for downsample (has blob/path+idx)
BOX="${BOX:-320}"
APIX="${APIX:-0.83}"
IMAGES_DIR="${IMAGES_DIR:?set IMAGES_DIR to the parent dir of J226/ (blob/path -> J226/reconstructed/<uid>_particles.mrc, same images as J264/J235)}"

POSES="${POSES:-$J399_DIR/inputs/poses.pkl}"
CTF="${CTF:-$J399_DIR/inputs/ctf.pkl}"

# Hyperparameters (validated sweet spot from J1442/J264 sweeps)
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

# subset (divide-and-conquer)
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
OUT="$J399_DIR/$TAG"
PARTICLES="${PARTICLES:-$J399_DIR/inputs/particles.${D}.mrcs}"
CLUSTERS="$OUT/cluster_exports_k${KCLUST}"
mkdir -p "$OUT" "$J399_DIR/inputs"

echo "=============================================================="
echo " J399 high-recall run with AUTO-RESUME & VALIDATION"
echo "   D=$D  zdim=$ZDIM  beta=$BETA  epochs=$EPOCHS"
echo "   enc=${ENC_DIM}x${ENC_LAYERS}  dec=${DEC_DIM}x${DEC_LAYERS}  batch=$BATCH"
echo "   out=$OUT"
echo "=============================================================="

# --------------------------------------------------------------------------- #
# 0) Derive poses.pkl / ctf.pkl if missing (J399 has no prior preprocessing)
# --------------------------------------------------------------------------- #
[[ -f "$MAIN" ]] || { echo "MISSING MAIN particles file: $MAIN" >&2; exit 1; }

if [[ -f "$POSES" ]]; then
  echo "[0/5] poses.pkl: reuse existing $POSES"
else
  echo "[0/5] parse_pose_csparc -> $POSES"
  "$CRYODRGN" parse_pose_csparc "$MAIN" -D "$BOX" -o "$POSES"
fi

if [[ -f "$CTF" ]]; then
  echo "[0/5] ctf.pkl: reuse existing $CTF"
else
  echo "[0/5] parse_ctf_csparc -> $CTF"
  "$CRYODRGN" parse_ctf_csparc "$MAIN" -D "$BOX" --Apix "$APIX" -o "$CTF"
fi

# --------------------------------------------------------------------------- #
# 1) Downsample particle stack to D (skip if present)
# --------------------------------------------------------------------------- #
if [[ -f "$PARTICLES" ]]; then
  echo "[1/5] downsample: reuse existing $PARTICLES"
else
  [[ -d "$IMAGES_DIR" ]] || { echo "IMAGES_DIR not a directory: $IMAGES_DIR" >&2; exit 1; }
  echo "[1/5] downsample -> $PARTICLES"
  "$CRYODRGN" downsample "$PASS" -D "$D" --datadir "$IMAGES_DIR" -o "$PARTICLES"
fi

# --------------------------------------------------------------------------- #
# 2) Find the latest VALID checkpoint
# --------------------------------------------------------------------------- #
echo "[2/5] Looking for valid checkpoint in $OUT"

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
  echo "[3/5] Training already complete. Skipping."
else
  echo "[3/5] train_vae -> $OUT"

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
# 5) analyze (optional)
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
  -k "$KCLUST" --dataset J399 \
  --min-resp 0.8 \
  -o "$CLUSTERS"

cat <<EOF

==============================================================
DONE.
==============================================================
EOF
