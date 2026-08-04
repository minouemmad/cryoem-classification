#!/usr/bin/env bash
###############################################################################
# FOCUSED cryoDRGN pipeline for J399 (hP7W2, J264 homorefine / true global
# consensus pose) — hudson. Mirrors focused_pipeline_j2708.sh / focused_pipeline.sh
# (built/validated for J1442). Same 4 steps:
#   1. find what MOVES   -> per-voxel variance across decoded states -> 3D mask
#   2. isolate it         -> pose-aware signal subtraction (cryodrgn_focus_subtract.py)
#   3. train focused      -> cryoDRGN on the residual stack (same poses/ctf)
#   4. recombine          -> cluster the sharper latent; labels apply to the
#                            ORIGINAL particles -> refine those per cluster
#
# Prereq: run scripts/cryodrgn/j399_recover_states.sh FIRST (the "sweet spot"
# zdim16/beta0.03/D128 run) so there's an analyze.<EP>/kmeans20/ to build the
# mask from. Default RUN below points at that run's output.
#
# USAGE (repo root on hudson, cryodrgn env):
#   export IMAGES_DIR=/path/to/parent/of/J226   # same images as J264/J235
#   bash scripts/cryodrgn/j399_recover_states.sh          # step 0: produce RUN
#   bash scripts/cryodrgn/focused_pipeline_j399.sh        # steps 1-4 below
###############################################################################
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python}"                                   # cryodrgn env python
J399_DIR="${J399_DIR:-$REPO/results_cryodrgn/J399_real}"

RUN="${RUN:-$J399_DIR/train_recover_D128_z16_b0p03}"   # from j399_recover_states.sh
EP="${EP:-50}"
APIX="${APIX:-2.075}"                                # J399: orig box320 @0.83A -> D128
VOLS="${VOLS:-$RUN/analyze.$EP/kmeans20/vol_*.mrc}"

PARTICLES="${PARTICLES:-$J399_DIR/inputs/particles.128.mrcs}"
POSES="${POSES:-$J399_DIR/inputs/poses.pkl}"
CTF="${CTF:-$J399_DIR/inputs/ctf.pkl}"

FOCUS_DIR="${FOCUS_DIR:-$J399_DIR/focus}"
RESID="${RESID:-$J399_DIR/inputs/particles.128.focus.mrcs}"
MASKQ="${MASKQ:-0.90}"; DILATE="${DILATE:-8}"; EDGE="${EDGE:-6}"

echo "== 0. sanity: subtraction geometry self-test =="
"$PY" "$REPO/scripts/cryodrgn/cryodrgn_focus_subtract.py" --self-test

echo "== 1. focus mask + consensus from decoded states =="
"$PY" "$REPO/scripts/cryodrgn/make_focus_mask.py" \
  --volumes "$VOLS" --apix "$APIX" \
  --mask-quantile "$MASKQ" --dilate "$DILATE" --edge "$EDGE" \
  -o "$FOCUS_DIR"
echo "   -> INSPECT $FOCUS_DIR/focus_mask.mrc over consensus.mrc in ChimeraX."
echo "      Re-run step 1 with different --mask-quantile/--dilate if it doesn't"
echo "      cover the moving domain."

echo "== 2. focused signal subtraction (2k-particle sanity first) =="
"$PY" "$REPO/scripts/cryodrgn/cryodrgn_focus_subtract.py" \
  --particles "$PARTICLES" --poses "$POSES" --ctf "$CTF" \
  --consensus "$FOCUS_DIR/consensus.mrc" --mask "$FOCUS_DIR/focus_mask.mrc" \
  --first 2000 -o "${RESID%.mrcs}.sanity.mrcs"
echo "   -> check the [fit] scale stats look sane (nonzero, not exploding), then"
echo "      the full run below. Comment out --first for all particles."
"$PY" "$REPO/scripts/cryodrgn/cryodrgn_focus_subtract.py" \
  --particles "$PARTICLES" --poses "$POSES" --ctf "$CTF" \
  --consensus "$FOCUS_DIR/consensus.mrc" --mask "$FOCUS_DIR/focus_mask.mrc" \
  -o "$RESID"

echo "== 3. train cryoDRGN on the focused (residual) stack =="
PARTICLES="$RESID" TAG="train_focus_D128_z16_b0p03" \
  ZDIM=16 BETA=0.03 D=128 EPOCHS=50 SEED=0 SKIP_EXPORT=1 \
  bash "$REPO/scripts/cryodrgn/j399_recover_states.sh"

echo "== 4. score the focused latent (did the mobile region split?) =="
"$PY" "$REPO/scripts/cryodrgn/cryodrgn_sweep_score.py" \
  --runs "$J399_DIR/train_focus_D128_z16_b0p03" \
  -o "$J399_DIR/train_focus_D128_z16_b0p03/focus_score" --sep-thresh 1.5

cat <<EOF

DONE.  If the focused latent shows more reproducible modes than the global
baseline, carve them (make_subset_ind.py on the FOCUSED z) or over-cluster +
export (export_latent_clusters.py) -> the labels apply to the ORIGINAL
particles -> CryoSPARC ab-initio -> NU on each.  Run a 2nd seed
(SEED=1 TAG=train_focus_D128_z16_b0p03_s1 ... ) to confirm any new split is
reproducible before committing GPU to refinement.  Compare against the J264
focused run (same images, biased poses) to see whether the pose-safe J399
frame changes which states separate.
EOF
