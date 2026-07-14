#!/usr/bin/env bash
# =============================================================================
# J2708 + J4624  cryoDRGN pipeline — junk-filter purified subset + fullset
# Paper: Zhong et al. 2021 Nature Methods (EMPIAR-10028 / Fig 5 workflow)
#
# Run on Hudson after scping data.  Assumes:
#   IMAGES_J2708  = directory on cluster that contains J2694/reconstructed/
#   IMAGES_J4624  = directory on cluster that contains J2014/reconstructed/
#   (set these to the parent CryoSPARC project directory on the cluster)
#
# Usage (from /home/mae2183/cryoem-classification/):
#   bash scripts/cryodrgn/run_j2708_j4624.sh 2>&1 | tee run_j2708_j4624.log
# =============================================================================
set -euo pipefail

REPO=/home/mae2183/cryoem-classification
CONDA_ENV=cryodrgn
CONDA_ACTIVATE="source ~/miniconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV"

# ── EDIT THESE ────────────────────────────────────────────────────────────────
IMAGES_J2708=/path/to/cryosparc/P30/projects    # must contain J2694/reconstructed/
IMAGES_J4624=/path/to/cryosparc/P37/projects    # must contain J2014/reconstructed/
# ──────────────────────────────────────────────────────────────────────────────

cd "$REPO"
eval "$CONDA_ACTIVATE"

# =============================================================================
# SHARED PARAMETERS
# =============================================================================
# J2708: N=231930, box=250, psize=1.075, 10 classes (6 dummy + 4 protein)
J2708_PASS_ALN="data/J2708/cryosparc_P30_J2708_passthrough_particles_all_classes_alignments3D.cs"
J2708_PASS_CTF="data/J2708/cryosparc_P30_J2708_passthrough_particles_all_classes_ctf.cs"
J2708_PASS_BLOB="data/J2708/cryosparc_P30_J2708_passthrough_particles_all_classes_blob.cs"
J2708_D_FULL=250
J2708_APIX_FULL=1.075
J2708_APIX_128=2.102   # 1.075 * 250/128
J2708_APIX_256=1.055   # 1.075 * 250/256 (used for closest power-of-2 full-res)

# J4624: N=304853, box=320, psize=0.83, 11 classes (6 dummy + 5 protein)
J4624_PASS_ALN="data/J4624/cryosparc_P37_J4624_passthrough_particles_all_classes_alignments3D.cs"
J4624_PASS_CTF="data/J4624/cryosparc_P37_J4624_passthrough_particles_all_classes_ctf.cs"
J4624_PASS_BLOB="data/J4624/cryosparc_P37_J4624_passthrough_particles_all_classes_blob.cs"
J4624_D_FULL=320
J4624_APIX_FULL=0.83
J4624_APIX_128=2.075   # 0.83 * 320/128
J4624_APIX_256=1.0375  # 0.83 * 320/256


# =============================================================================
# STEP 1 — PREPROCESSING (parse poses + CTF, downsample)
# =============================================================================
echo "====== STEP 1: PREPROCESSING ======"

# ── J2708 ──────────────────────────────────────────────────────────────────
mkdir -p results_cryodrgn/J2708/inputs

echo "[J2708] parse poses..."
cryodrgn parse_pose_csparc "$J2708_PASS_ALN" \
    -D $J2708_D_FULL \
    -o results_cryodrgn/J2708/inputs/poses.pkl

echo "[J2708] parse CTF..."
cryodrgn parse_ctf_csparc "$J2708_PASS_CTF" \
    -D $J2708_D_FULL \
    --Apix $J2708_APIX_FULL \
    -o results_cryodrgn/J2708/inputs/ctf.pkl

echo "[J2708] downsample D=128..."
cryodrgn downsample "$J2708_PASS_BLOB" \
    -D 128 \
    --datadir "$IMAGES_J2708" \
    -o results_cryodrgn/J2708/inputs/particles.128.mrcs

# Full-res: J2708 native box is 250; keep at 250 (no downsample needed)
# Just symlink/copy the blob reference for training
echo "[J2708] downsample D=$J2708_D_FULL (full res)..."
cryodrgn downsample "$J2708_PASS_BLOB" \
    -D $J2708_D_FULL \
    --datadir "$IMAGES_J2708" \
    -o results_cryodrgn/J2708/inputs/particles.250.mrcs

# ── J4624 ──────────────────────────────────────────────────────────────────
mkdir -p results_cryodrgn/J4624/inputs

echo "[J4624] parse poses..."
cryodrgn parse_pose_csparc "$J4624_PASS_ALN" \
    -D $J4624_D_FULL \
    -o results_cryodrgn/J4624/inputs/poses.pkl

echo "[J4624] parse CTF..."
cryodrgn parse_ctf_csparc "$J4624_PASS_CTF" \
    -D $J4624_D_FULL \
    --Apix $J4624_APIX_FULL \
    -o results_cryodrgn/J4624/inputs/ctf.pkl

echo "[J4624] downsample D=128..."
cryodrgn downsample "$J4624_PASS_BLOB" \
    -D 128 \
    --datadir "$IMAGES_J4624" \
    -o results_cryodrgn/J4624/inputs/particles.128.mrcs

echo "[J4624] downsample D=256..."
cryodrgn downsample "$J4624_PASS_BLOB" \
    -D 256 \
    --datadir "$IMAGES_J4624" \
    -o results_cryodrgn/J4624/inputs/particles.256.mrcs


# =============================================================================
# STEP 2 — JUNK FILTER PILOT  (z=1 model, D=128, 50 epochs)
#   Paper Fig 5: train 1-D latent to separate junk from protein
#   Junk = particles whose z_1 < -1 (low-frequency / aggregated / broken)
# =============================================================================
echo "====== STEP 2: JUNK FILTER PILOTS (z=1, D=128, 50ep) ======"

# ── J2708 z=1 pilot ────────────────────────────────────────────────────────
mkdir -p results_cryodrgn/J2708/pilot_z1
cryodrgn train_vae results_cryodrgn/J2708/inputs/particles.128.mrcs \
    --poses  results_cryodrgn/J2708/inputs/poses.pkl \
    --ctf    results_cryodrgn/J2708/inputs/ctf.pkl \
    --zdim 1 \
    --enc-dim 256 --enc-layers 3 \
    --dec-dim 256 --dec-layers 3 \
    --no-amp -n 50 \
    --lazy --num-workers 4 --batch-size 64 \
    -o results_cryodrgn/J2708/pilot_z1

# Filter: keep particles where z_1 >= -1 (remove junk outliers)
python - <<'PYEOF'
import pickle, numpy as np
z = pickle.load(open('results_cryodrgn/J2708/pilot_z1/z.50.pkl','rb')).squeeze()
ind_keep = np.where(z >= -1)[0]
ind_rem  = np.where(z <  -1)[0]
with open('results_cryodrgn/J2708/inputs/ind_keep.pkl','wb') as f: pickle.dump(ind_keep, f)
with open('results_cryodrgn/J2708/inputs/ind_removed.pkl','wb') as f: pickle.dump(ind_rem, f)
print(f'J2708 junk filter: kept {len(ind_keep)}/{len(z)}  removed {len(ind_rem)}')
PYEOF
# NOTE: np.save for .pkl must match cryodrgn's expected format
# cryodrgn uses np.load on the ind file; the above saves a plain ndarray

# ── J4624 z=1 pilot ────────────────────────────────────────────────────────
mkdir -p results_cryodrgn/J4624/pilot_z1
cryodrgn train_vae results_cryodrgn/J4624/inputs/particles.128.mrcs \
    --poses  results_cryodrgn/J4624/inputs/poses.pkl \
    --ctf    results_cryodrgn/J4624/inputs/ctf.pkl \
    --zdim 1 \
    --enc-dim 256 --enc-layers 3 \
    --dec-dim 256 --dec-layers 3 \
    --no-amp -n 50 \
    --lazy --num-workers 4 --batch-size 64 \
    -o results_cryodrgn/J4624/pilot_z1

python - <<'PYEOF'
import pickle, numpy as np
z = pickle.load(open('results_cryodrgn/J4624/pilot_z1/z.50.pkl','rb')).squeeze()
ind_keep = np.where(z >= -1)[0]
ind_rem  = np.where(z <  -1)[0]
with open('results_cryodrgn/J4624/inputs/ind_keep.pkl','wb') as f: pickle.dump(ind_keep, f)
with open('results_cryodrgn/J4624/inputs/ind_removed.pkl','wb') as f: pickle.dump(ind_rem, f)
print(f'J4624 junk filter: kept {len(ind_keep)}/{len(z)}  removed {len(ind_rem)}')
PYEOF


# =============================================================================
# STEP 3 — PURIFIED SUBSET FINAL TRAINING  (z=10, D=full, 100 epochs)
#   Uses ind_keep.pkl to exclude junk identified in Step 2
#   Paper: D=256, 1024×3, zdim=10 — "high-resolution final model"
# =============================================================================
echo "====== STEP 3: PURIFIED SUBSET TRAINING (D=full, z=10, 100ep) ======"

# ── J2708 purified (D=250) ─────────────────────────────────────────────────
mkdir -p results_cryodrgn/J2708/purified_D250_z10
cryodrgn train_vae results_cryodrgn/J2708/inputs/particles.250.mrcs \
    --poses results_cryodrgn/J2708/inputs/poses.pkl \
    --ctf   results_cryodrgn/J2708/inputs/ctf.pkl \
    --ind   results_cryodrgn/J2708/inputs/ind_keep.pkl \
    --zdim 10 \
    --enc-dim 1024 --enc-layers 3 \
    --dec-dim 1024 --dec-layers 3 \
    --no-amp -n 100 \
    --lazy --num-workers 4 --batch-size 64 \
    --checkpoint 1 --log-interval 5000 \
    -o results_cryodrgn/J2708/purified_D250_z10

# ── J4624 purified (D=256) ─────────────────────────────────────────────────
mkdir -p results_cryodrgn/J4624/purified_D256_z10
cryodrgn train_vae results_cryodrgn/J4624/inputs/particles.256.mrcs \
    --poses results_cryodrgn/J4624/inputs/poses.pkl \
    --ctf   results_cryodrgn/J4624/inputs/ctf.pkl \
    --ind   results_cryodrgn/J4624/inputs/ind_keep.pkl \
    --zdim 10 \
    --enc-dim 1024 --enc-layers 3 \
    --dec-dim 1024 --dec-layers 3 \
    --no-amp -n 100 \
    --lazy --num-workers 4 --batch-size 64 \
    --checkpoint 1 --log-interval 5000 \
    -o results_cryodrgn/J4624/purified_D256_z10


# =============================================================================
# STEP 4 — FULLSET TRAINING  (z=10, D=full, 100 epochs, no junk filter)
#   No --ind flag: trains on all particles
# =============================================================================
echo "====== STEP 4: FULLSET TRAINING (D=full, z=10, 100ep) ======"

# ── J2708 fullset ──────────────────────────────────────────────────────────
mkdir -p results_cryodrgn/J2708/fullset_D250_z10
cryodrgn train_vae results_cryodrgn/J2708/inputs/particles.250.mrcs \
    --poses results_cryodrgn/J2708/inputs/poses.pkl \
    --ctf   results_cryodrgn/J2708/inputs/ctf.pkl \
    --zdim 10 \
    --enc-dim 1024 --enc-layers 3 \
    --dec-dim 1024 --dec-layers 3 \
    --no-amp -n 100 \
    --lazy --num-workers 4 --batch-size 64 \
    --checkpoint 1 --log-interval 5000 \
    -o results_cryodrgn/J2708/fullset_D250_z10

# ── J4624 fullset ──────────────────────────────────────────────────────────
mkdir -p results_cryodrgn/J4624/fullset_D256_z10
cryodrgn train_vae results_cryodrgn/J4624/inputs/particles.256.mrcs \
    --poses results_cryodrgn/J4624/inputs/poses.pkl \
    --ctf   results_cryodrgn/J4624/inputs/ctf.pkl \
    --zdim 10 \
    --enc-dim 1024 --enc-layers 3 \
    --dec-dim 1024 --dec-layers 3 \
    --no-amp -n 100 \
    --lazy --num-workers 4 --batch-size 64 \
    --checkpoint 1 --log-interval 5000 \
    -o results_cryodrgn/J4624/fullset_D256_z10


# =============================================================================
# STEP 5 — ANALYSIS  (analyze at epoch 50 and 100 for each run)
#   --ksample 20 = generate 20 k-means cluster volumes
#   --pc 3       = generate PC1, PC2, PC3 traversal z-files
# =============================================================================
echo "====== STEP 5: ANALYSIS ======"

for DS in J2708 J4624; do
    for RUN in purified fullset; do
        if [ "$DS" == "J2708" ]; then
            WDIR="results_cryodrgn/${DS}/${RUN}_D250_z10"
        else
            WDIR="results_cryodrgn/${DS}/${RUN}_D256_z10"
        fi

        # Analyze at epoch 50 (mid-training checkpoint)
        if [ -f "${WDIR}/z.50.pkl" ]; then
            echo "[${DS} ${RUN}] analyze ep50..."
            cryodrgn analyze "$WDIR" 50 \
                --ksample 20 --pc 3 \
                -o "${WDIR}/analyze.50"
        fi

        # Analyze at epoch 100 (final)
        if [ -f "${WDIR}/z.100.pkl" ]; then
            echo "[${DS} ${RUN}] analyze ep100..."
            cryodrgn analyze "$WDIR" 100 \
                --ksample 20 --pc 3 \
                -o "${WDIR}/analyze.100"
        fi
    done
done

echo "====== ALL STEPS COMPLETE ======"
echo "Results are in:"
echo "  results_cryodrgn/J2708/purified_D250_z10/"
echo "  results_cryodrgn/J2708/fullset_D250_z10/"
echo "  results_cryodrgn/J4624/purified_D256_z10/"
echo "  results_cryodrgn/J4624/fullset_D256_z10/"
