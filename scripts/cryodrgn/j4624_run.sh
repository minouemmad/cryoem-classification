#!/usr/bin/env bash
###############################################################################
# J4624 cryoDRGN run  (hudson / GPU)  --  full preprocess -> train -> analyze
#
# J4624 (CryoSPARC project P37): 304,853 particles, box 320, 0.83 A/pix,
# blob/sign -1.0.  11 hetero classes = 6 dummy + 5 protein (protein-idx 6..10,
# n-dummies 6; argmax fractions ~ [.,.,.,.,.,., .185 .117 .420 .161 .095]).
# Images (blob/path) -> J2014/reconstructed/<uid>_particles.mrc, so IMAGES_DIR
# must be the PARENT directory of J2014/ on hudson.
#
# Unlike J207, J4624 already carries SINGLE consensus poses in its passthrough
# alignments3D .cs, so we parse poses directly (no build_consensus_pose_cs step).
#
# Trains on the FULLSET (no --ind): the J264 purified-vs-fullset test showed the
# junk-filtered subset separated classes WORSE, so purification is skipped.
#
# ---------------------------------------------------------------------------
# USAGE (on hudson, cryodrgn env, from repo root /home/mae2183/cryoem-classification):
#   export IMAGES_DIR=/path/to/dir/that/contains/BOTH/J2014/AND/J2015/
#     REQUIRED.  J4624's images are SPLIT across two Reference-Based Motion
#     Correction jobs: blob/path -> 'J2014/reconstructed/...' (81,429 particles)
#     AND 'J2015/reconstructed/...' (223,424).  So IMAGES_DIR must be the PARENT
#     directory that contains both J2014/ and J2015/ (the CryoSPARC project dir);
#     cryoDRGN then resolves each blob/path via <datadir>/<blob/path> natively.
#     Do NOT point at a single reconstructed/ folder (there are two).
#   bash scripts/cryodrgn/j4624_run.sh              # D=256 final by default
#
# Cheap pilot first (recommended) -- D=128, small net, 50 ep:
#   D=128 ENC_DIM=256 DEC_DIM=256 TAG=pilot_z10_D128 bash scripts/cryodrgn/j4624_run.sh
#
# Override any hyperparameter via env, e.g.:
#   ZDIM=10 EPOCHS=50 BATCH=64 bash scripts/cryodrgn/j4624_run.sh
#
# NOTE: 304k particles at D=256 is SLOW (hours/epoch on the current HW). Do the
# D=128 pilot first; only commit to the D=256 final if the pilot shows structure.
###############################################################################
set -euo pipefail

# --------------------------------------------------------------------------- #
# Config (override via environment variables)
# --------------------------------------------------------------------------- #
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CRYODRGN="${CRYODRGN:-cryodrgn}"

J4624_DIR="${J4624_DIR:-$REPO/results_cryodrgn/J4624_real}"
DATA="${DATA:-$REPO/data/J4624}"

# Inputs (CryoSPARC field-group exports already present under data/J4624)
BLOB="${BLOB:-$DATA/cryosparc_P37_J4624_passthrough_particles_all_classes_blob.cs}"
ALIGN="${ALIGN:-$DATA/cryosparc_P37_J4624_passthrough_particles_all_classes_alignments3D.cs}"
CTF_CS="${CTF_CS:-$DATA/cryosparc_P37_J4624_passthrough_particles_all_classes_ctf.cs}"
IMAGES_DIR="${IMAGES_DIR:?set IMAGES_DIR to the PARENT dir that contains BOTH J2014/ and J2015/ (images are split across two RBMC jobs: J2014/reconstructed/... + J2015/reconstructed/...). Point at the project dir holding both, NOT a single reconstructed/ folder.}"

# Acquisition geometry
BOX="${BOX:-320}"             # original image box (for parse_pose/parse_ctf)
APIX="${APIX:-0.83}"          # original pixel size

# Training hyperparameters (defaults = paper-style D=256 final)
D="${D:-256}"                 # encoder/downsample box
ZDIM="${ZDIM:-10}"
EPOCHS="${EPOCHS:-50}"        # 50 is enough; loss converges by ~ep25-35 for this data
ENC_DIM="${ENC_DIM:-1024}"
ENC_LAYERS="${ENC_LAYERS:-3}"
DEC_DIM="${DEC_DIM:-1024}"
DEC_LAYERS="${DEC_LAYERS:-3}"
BATCH="${BATCH:-64}"
NWORKERS="${NWORKERS:-4}"

# Output layout
TAG="${TAG:-train_final_z${ZDIM}_D${D}}"
OUT="$J4624_DIR/$TAG"
INPUTS="$J4624_DIR/inputs"
POSES="${POSES:-$INPUTS/poses.pkl}"
CTF="${CTF:-$INPUTS/ctf.pkl}"
# With --chunk, `downsample -o particles.$D.mrcs` writes particles.$D.0.mrcs,
# particles.$D.1.mrcs, ... plus a particles.$D.txt index. train_vae is fed the
# .txt (it lists the chunks); a single particles.$D.mrcs is NEVER created.
PARTICLES_BASE="${PARTICLES_BASE:-$INPUTS/particles.${D}}"
PARTICLES_TXT="${PARTICLES_BASE}.txt"
# decoded-map apix for later eval_vol: APIX*BOX/D
APIX_D="$(python -c "print(round($APIX*$BOX/$D,4))")"

mkdir -p "$OUT" "$INPUTS"

echo "=============================================================="
echo " J4624 cryoDRGN run"
echo "   particles=304,853  box=$BOX  apix=$APIX  (decoded apix@D=$D -> $APIX_D)"
echo "   D=$D  zdim=$ZDIM  epochs=$EPOCHS  enc=${ENC_DIM}x${ENC_LAYERS}  dec=${DEC_DIM}x${DEC_LAYERS}"
echo "   IMAGES_DIR=$IMAGES_DIR"
echo "   out=$OUT"
echo "=============================================================="

# --------------------------------------------------------------------------- #
# 0) Sanity checks
# --------------------------------------------------------------------------- #
for f in "$BLOB" "$ALIGN" "$CTF_CS"; do
  [[ -f "$f" ]] || { echo "MISSING input: $f" >&2; exit 1; }
done
[[ -d "$IMAGES_DIR" ]] || { echo "IMAGES_DIR not a directory: $IMAGES_DIR" >&2; exit 1; }

# --------------------------------------------------------------------------- #
# 1) Poses  (single consensus poses from the passthrough alignments3D; box=$BOX)
# --------------------------------------------------------------------------- #
if [[ -f "$POSES" ]]; then
  echo "[1/5] poses: reuse existing $POSES"
else
  echo "[1/5] parse_pose_csparc -> $POSES"
  "$CRYODRGN" parse_pose_csparc "$ALIGN" -D "$BOX" -o "$POSES"
fi

# --------------------------------------------------------------------------- #
# 2) CTF  (box=$BOX, Apix=$APIX)
# --------------------------------------------------------------------------- #
if [[ -f "$CTF" ]]; then
  echo "[2/5] ctf: reuse existing $CTF"
else
  echo "[2/5] parse_ctf_csparc -> $CTF"
  "$CRYODRGN" parse_ctf_csparc "$CTF_CS" -D "$BOX" --Apix "$APIX" -o "$CTF"
fi

# --------------------------------------------------------------------------- #
# 3) Downsample images to D  (--datadir = parent containing BOTH J2014/ & J2015/;
#    with --chunk this writes particles.$D.{0,1,...}.mrcs + particles.$D.txt)
# --------------------------------------------------------------------------- #
if [[ -f "$PARTICLES_TXT" ]]; then
  echo "[3/5] downsample: reuse existing $PARTICLES_TXT"
else
  echo "[3/5] downsample -> ${PARTICLES_BASE}.*.mrcs (+ $PARTICLES_TXT)"
  "$CRYODRGN" downsample "$BLOB" -D "$D" --datadir "$IMAGES_DIR" \
    --chunk 50000 -o "${PARTICLES_BASE}.mrcs"
fi

# --------------------------------------------------------------------------- #
# 4) train_vae  (FULLSET: no --ind.  --no-amp avoids fp16 NaN; --lazy for RAM)
#    Input = the .txt index that lists the downsampled chunks.
# --------------------------------------------------------------------------- #
echo "[4/5] train_vae -> $OUT"
"$CRYODRGN" train_vae "$PARTICLES_TXT" \
  --ctf "$CTF" --poses "$POSES" \
  --zdim "$ZDIM" --num-epochs "$EPOCHS" \
  --enc-dim "$ENC_DIM" --enc-layers "$ENC_LAYERS" \
  --dec-dim "$DEC_DIM" --dec-layers "$DEC_LAYERS" \
  --batch-size "$BATCH" --lazy --num-workers "$NWORKERS" --no-amp \
  --checkpoint 1 -o "$OUT"

# newest z.<N>.pkl / epoch
ZFINAL="$(ls -1 "$OUT"/z.*.pkl 2>/dev/null | sed -E 's/.*z\.([0-9]+)\.pkl/\1 &/' \
          | sort -n | tail -1 | awk '{print $2}')"
[[ -n "${ZFINAL:-}" ]] || { echo "no z.*.pkl produced" >&2; exit 1; }
EP="$(basename "$ZFINAL" | sed -E 's/z\.([0-9]+)\.pkl/\1/')"
echo "      final latent: $ZFINAL (epoch $EP)"

# --------------------------------------------------------------------------- #
# 5) analyze  (UMAP/PCA + kmeans volumes + PC traversals)
# --------------------------------------------------------------------------- #
echo "[5/5] analyze epoch $EP (--Apix $APIX_D)"
"$CRYODRGN" analyze "$OUT" "$EP" --Apix "$APIX_D" \
  || echo "  (analyze failed/optional -- continuing)"

cat <<EOF

==============================================================
DONE (J4624, $TAG).  Pull the results dir back to the workstation, then run
the LOCAL analysis (cryodrgn-py310) — J4624 is a 5-protein-class hetero:
  --protein-idx 6,7,8,9,10  --n-dummies 6
Suggested local follow-ups:
  * scripts/cryodrgn/cryodrgn_conformational_landscape.py  (PC densities / class overlay)
  * scripts/cryodrgn/cryodrgn_latent_gmm.py                (soft populations + uncertainty)
  * scripts/cryodrgn/cryodrgn_run_separation_compare.py    (if you also do a pilot vs final)
CryoSPARC .cs for the class overlay:
  data/J4624/cryosparc_P37_J4624_00062_particles_alignments3D_multi.cs
Passthrough (z row order) for uid alignment:
  $BLOB
==============================================================
EOF
