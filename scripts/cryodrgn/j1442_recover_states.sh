#!/usr/bin/env bash
###############################################################################
# J1442 "high-recall" state-recovery run  (hudson / GPU)
#
# GOAL: recover MORE / subtler intermediate states in the cryoDRGN latent so the
# downstream (cluster -> export -> CryoSPARC ab-initio -> NU) pipeline can pull
# out the classes you're currently missing (target ~5 for J1442).  You said you
# would rather over-cluster than under-cluster; this run is tuned for exactly
# that -- HIGH RECALL -- and lets CryoSPARC ab-initio arbitrate afterwards.
#
# ---------------------------------------------------------------------------
# IS THE MAIN ALGORITHM INTACT?  YES.
# Every training change below is a STOCK `cryodrgn train_vae` command-line flag
# (see cryodrgn/cryodrgn/commands/train_vae.py).  No source code is modified.
# Same VAE, same loss, same encoder/decoder.  We only:
#   (1) lower the KL weight  (--beta): the KL term pulls z toward one isotropic
#       Gaussian; that is what smooths distinct states together.  Lower beta =>
#       states allowed to spread into separable clusters (default was 1/zdim).
#   (2) raise latent capacity (--zdim 16) so >3 independent motions have room.
#   (3) encode at D=128 (not 256): your own runs showed D=128 -> 3 clean basins
#       while D=256 collapsed to 1 continuous blob (high-freq noise dominates the
#       latent at high res).  D=128 is the sensitivity sweet spot for SORTING.
#   (4) shrink the decoder (--dec-dim 512): you never use the decoder maps, so
#       spend the compute on the encoder + epochs, not high-res reconstruction.
# The clustering/export step is post-hoc analysis of z.pkl -- it never touches
# cryoDRGN.  Nothing here changes the algorithm.
# ---------------------------------------------------------------------------
#
# USAGE (on hudson, in the cryodrgn env, from the repo root
#        /home/mae2183/cryoem-classification):
#   export IMAGES_DIR=/path/to/parent/of/blob-referenced/.mrc     # REQUIRED
#   bash scripts/cryodrgn/j1442_recover_states.sh
#
# Override any hyperparameter without editing the file, e.g.:
#   ZDIM=20 BETA=0.02 D=128 EPOCHS=50 KCLUST=12 bash scripts/cryodrgn/j1442_recover_states.sh
#
# Paths below follow the hudson layout: cryoDRGN artifacts for J1442 live under
# results_cryodrgn/J1442_real/ (alongside inputs/, pilot_z10/, train_fullset/).
###############################################################################
set -euo pipefail

# --------------------------------------------------------------------------- #
# Config (override via environment variables)
# --------------------------------------------------------------------------- #
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"   # = /home/mae2183/cryoem-classification on hudson
PYCLUSTER="${PYCLUSTER:-python}"            # cryodrgn-py310 python for the export step
CRYODRGN="${CRYODRGN:-cryodrgn}"           # cryodrgn executable

# Base dir for J1442 cryoDRGN artifacts (hudson convention: J1442_real)
J1442_DIR="${J1442_DIR:-$REPO/results_cryodrgn/J1442_real}"

# Inputs (already produced on hudson; poses/ctf are convention-correct)
PASS="${PASS:-$REPO/data/cryosparc_P25_J1442_passthrough_particles_all_classes.cs}"
POSES="${POSES:-$J1442_DIR/inputs/poses.pkl}"
CTF="${CTF:-$J1442_DIR/inputs/ctf.pkl}"
IMAGES_DIR="${IMAGES_DIR:?set IMAGES_DIR to the parent dir of the blob/path .mrc image stacks (J1442 blob/path -> J995/reconstructed/<uid>_particles.mrc)}"

# Tuned "high-recall" hyperparameters
D="${D:-128}"                 # encoder box (128 = sensitivity sweet spot for sorting)
ZDIM="${ZDIM:-16}"            # latent dim (room for >3 independent motions)
BETA="${BETA:-0.03}"          # KL weight (default was 1/zdim = 0.0625 at zdim16; 0.03 relaxes it)
EPOCHS="${EPOCHS:-50}"        # 50 is plenty; loss converges by ~ep10-25 for this data
ENC_DIM="${ENC_DIM:-1024}"
ENC_LAYERS="${ENC_LAYERS:-3}"
DEC_DIM="${DEC_DIM:-512}"     # decoder unused downstream -> keep small/fast
DEC_LAYERS="${DEC_LAYERS:-3}"
BATCH="${BATCH:-64}"
NWORKERS="${NWORKERS:-4}"
SEED="${SEED:-0}"          # train_vae RNG seed (vary for reproducibility check in a sweep)
KCLUST="${KCLUST:-10}"        # OVER-cluster (target ~5): let ab-initio merge duplicates

# Output layout (under the hudson J1442_real dir, alongside train_fullset/ etc.)
TAG="${TAG:-train_recover_D${D}_z${ZDIM}_b${BETA//./p}}"
OUT="$J1442_DIR/$TAG"
PARTICLES="${PARTICLES:-$J1442_DIR/particles.${D}.mrcs}"   # shared downsampled stack (reused across runs)
CLUSTERS="$OUT/cluster_exports_k${KCLUST}"
mkdir -p "$OUT"

echo "=============================================================="
echo " J1442 high-recall run"
echo "   D=$D  zdim=$ZDIM  beta=$BETA  epochs=$EPOCHS"
echo "   enc=${ENC_DIM}x${ENC_LAYERS}  dec=${DEC_DIM}x${DEC_LAYERS}  batch=$BATCH"
echo "   over-cluster k=$KCLUST"
echo "   out=$OUT"
echo "=============================================================="

# --------------------------------------------------------------------------- #
# 0) Sanity checks
# --------------------------------------------------------------------------- #
for f in "$PASS" "$POSES" "$CTF"; do
  [[ -f "$f" ]] || { echo "MISSING input: $f" >&2; exit 1; }
done
[[ -d "$IMAGES_DIR" ]] || { echo "IMAGES_DIR not a directory: $IMAGES_DIR" >&2; exit 1; }

# --------------------------------------------------------------------------- #
# 1) Downsample particle stack to D (skip if present)
#    --datadir resolves the blob/path .mrc image stacks referenced by the .cs
# --------------------------------------------------------------------------- #
if [[ -f "$PARTICLES" ]]; then
  echo "[1/4] downsample: reuse existing $PARTICLES"
else
  echo "[1/4] downsample -> $PARTICLES"
  "$CRYODRGN" downsample "$PASS" -D "$D" --datadir "$IMAGES_DIR" -o "$PARTICLES"
fi

# --------------------------------------------------------------------------- #
# 2) train_vae  (FULLSET: no --ind; fullset separated classes better than the
#    junk-filtered subset in your prior J264 comparison)
# --------------------------------------------------------------------------- #
echo "[2/4] train_vae -> $OUT"
"$CRYODRGN" train_vae "$PARTICLES" \
  --ctf "$CTF" --poses "$POSES" \
  --zdim "$ZDIM" --beta "$BETA" --num-epochs "$EPOCHS" --seed "$SEED" \
  --enc-dim "$ENC_DIM" --enc-layers "$ENC_LAYERS" \
  --dec-dim "$DEC_DIM" --dec-layers "$DEC_LAYERS" \
  --batch-size "$BATCH" --lazy --num-workers "$NWORKERS" --no-amp \
  -o "$OUT"

# newest z.<N>.pkl
ZFINAL="$(ls -1 "$OUT"/z.*.pkl 2>/dev/null | sed -E 's/.*z\.([0-9]+)\.pkl/\1 &/' \
          | sort -n | tail -1 | awk '{print $2}')"
[[ -n "${ZFINAL:-}" ]] || { echo "no z.*.pkl produced" >&2; exit 1; }
echo "      final latent: $ZFINAL"

# --------------------------------------------------------------------------- #
# 3) analyze (UMAP/PCA + kmeans volumes; optional -- set SKIP_ANALYZE=1 in a sweep)
# --------------------------------------------------------------------------- #
EP="$(basename "$ZFINAL" | sed -E 's/z\.([0-9]+)\.pkl/\1/')"
if [[ "${SKIP_ANALYZE:-0}" == "1" ]]; then
  echo "[3/4] analyze: SKIPPED (SKIP_ANALYZE=1)"
else
  echo "[3/4] analyze epoch $EP"
  "$CRYODRGN" analyze "$OUT" "$EP" || echo "  (analyze failed/optional -- continuing)"
fi

# --------------------------------------------------------------------------- #
# 4) OVER-CLUSTER the FULL latent (all zdim, standardized) -> importable .cs
#    per cluster.  k=$KCLUST > target so genuine subtle states survive; run
#    ab-initio on each and merge duplicates by map comparison afterwards.
#    (optional -- set SKIP_EXPORT=1 in a sweep; do the export only on winners)
# --------------------------------------------------------------------------- #
if [[ "${SKIP_EXPORT:-0}" == "1" ]]; then
  echo "[4/4] over-cluster export: SKIPPED (SKIP_EXPORT=1)"
  echo "      latent ready: $ZFINAL"
  exit 0
fi
echo "[4/4] over-cluster (full z, k=$KCLUST) -> $CLUSTERS"
"$PYCLUSTER" "$REPO/scripts/cryodrgn/export_latent_clusters.py" \
  --z "$ZFINAL" \
  --passthrough-cs "$PASS" \
  -k "$KCLUST" --dataset J1442 \
  --min-resp 0.8 \
  -o "$CLUSTERS"

cat <<EOF

==============================================================
DONE.  Next steps (CryoSPARC):
  1. Import each $CLUSTERS/J1442_cluster_c*.cs  (blob+CTF present).
  2. Ab-initio (K=1) -> NU-refine each cluster.  Ab-initio is unbiased
     (Punjani 2017), so over-clustering does NOT bias the maps.
  3. Merge duplicates: compare the NU maps with
       python scripts/cryodrgn/compare_maps.py ...
     (rigid-align first -- independent refinements sit in their own pose
     frames, ~180 deg offsets are normal).  Clusters that converge to the
     same map = one state; distinct maps = your recovered intermediates.
  4. Report the MERGED, ab-initio-confirmed states (k=$KCLUST is recall,
     not the final state count).

To push harder on recall, re-run with e.g. BETA=0.02 ZDIM=20 KCLUST=14.
==============================================================
EOF
