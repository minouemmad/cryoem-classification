#!/usr/bin/env bash
###############################################################################
# FOCUSED cryoDRGN pipeline (hudson) — higher sensitivity to a moving region
#
# Implements focused heterogeneity INSIDE cryoDRGN (no CryoSPARC):
#   1. find what MOVES   -> per-voxel variance across decoded states -> 3D mask
#   2. isolate it         -> subtract the projected density OUTSIDE the mask from
#                            every particle, pose-aware, in cryoDRGN's own
#                            Hartley/CTF forward model (cryodrgn_focus_subtract.py)
#   3. train focused      -> cryoDRGN on the residual stack (same poses/ctf)
#   4. recombine          -> cluster the sharper latent; labels apply to the
#                            ORIGINAL particles -> refine those per cluster
#
# All conventions (pose coords@rot, compute_ctf, Apix rescale, translate_ht) are
# taken directly from cryoDRGN, and the geometry is unit-tested:
#   python scripts/cryodrgn/cryodrgn_focus_subtract.py --self-test   # must PASS
#
# USAGE (repo root on hudson, cryodrgn env):
#   RUN=results_cryodrgn/J1442_gP25_WT_POSE_BIAS/train_recover_D128_z16_b0p03 \
#   EP=50 APIX=2.075 \
#     bash scripts/cryodrgn/focused_pipeline.sh
###############################################################################
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python}"                                   # cryodrgn env python
J1442_DIR="${J1442_DIR:-$REPO/results_cryodrgn/J1442_gP25_WT_POSE_BIAS}"

RUN="${RUN:-$J1442_DIR/train_recover_D128_z16_b0p03}"   # a trained run w/ analyze
EP="${EP:-50}"
APIX="${APIX:-2.075}"                                # J1442 D=128 pixel size
# volumes that span the motion (kmeans20 from analyze, or PC traversal vols)
VOLS="${VOLS:-$RUN/analyze.$EP/kmeans20/vol_*.mrc}"

PARTICLES="${PARTICLES:-$J1442_DIR/inputs/particles.128.mrcs}"
POSES="${POSES:-$J1442_DIR/inputs/poses.pkl}"
CTF="${CTF:-$J1442_DIR/inputs/ctf.pkl}"

FOCUS_DIR="${FOCUS_DIR:-$J1442_DIR/focus}"
RESID="${RESID:-$J1442_DIR/inputs/particles.128.focus.mrcs}"
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
echo "      cover the moving domain (e.g. NBD1/NBD2/ICL4)."

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
  bash "$REPO/scripts/cryodrgn/j1442_recover_states.sh"

echo "== 4. score the focused latent (did the mobile region split?) =="
"$PY" "$REPO/scripts/cryodrgn/cryodrgn_sweep_score.py" \
  --runs "$J1442_DIR/train_focus_D128_z16_b0p03" \
  -o "$J1442_DIR/train_focus_D128_z16_b0p03/focus_score" --sep-thresh 1.5

cat <<EOF

DONE.  If the focused latent shows more reproducible modes than the global 3,
carve them (make_subset_ind.py on the FOCUSED z) or over-cluster + export
(export_latent_clusters.py) -> the labels apply to the ORIGINAL particles ->
CryoSPARC ab-initio -> NU on each.  Run a 2nd seed to confirm any new split is
reproducible before committing GPU to refinement.
EOF
