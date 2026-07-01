# J264 cryoDRGN runbook — following Zhong et al. (Nat. Methods 2021)

Follows the **Fig. 5 / EMPIAR-10076 assembly-state** workflow from Zhong et al.
(*Nature Methods* **18**, 176–185, 2021): pilot low-res runs to filter junk,
then a high-res final run to resolve metastable states, then a free-energy /
basin-occupancy read-out to decide *which* hetero-refinement jobs (and what K)
to run next.

- **Cluster:** `mae2183@hudson.biology.columbia.edu`, repo root
  `/home/mae2183/cryoem-classification/`, conda env `cryodrgn`
  (`/home/mae2183/miniconda3/envs/cryodrgn/bin/cryodrgn`)
- **Local workspace:** `C:\Users\maemm\OneDrive\Desktop\CryoEM\` (this repo,
  synced via git push/pull)
- **J264 particles:** 301,770 particles, box 320, 0.83 Å/pix, project **P7**,
  hetero-refine into 15 classes (6 dummy + **9 protein**, indices 6–14)

---

## Why this dataset

J264 is a fresh, larger 9-class hetero-refinement (301,770 particles). The whole
study's recurring finding is that **CryoSPARC structural classes collapse into
fewer cryoDRGN energetic basins** (J1442: 3 classes → 3 basins; J1497: 5 classes
→ 2–3 basins). The goal here is to test the same hypothesis on J264: does
cryoDRGN resolve **fewer than 9** metastable regions? If so, the basin-occupancy
matrix tells us exactly which of the 9 classes share a basin — i.e. which
particles to merge and what K to run in a focused follow-up hetero-refine.

---

## File inventory — what you have

Main files in `data/J264/`:

| file | rows | has blob | has 3D poses | has CTF |
|---|---|---|---|---|
| `cryosparc_P7_J264_00062_particles_alignments3D_multi.cs` | 301,770 | ✗ | `alignments3D_multi/*` (15-class, used to derive poses — see Step 1) | ✗ |
| `cryosparc_P7_J264_passthrough_particles_all_classes_blob.cs` | 301,770 | ✓ `J226/reconstructed/<uid>_particles.mrc`, D=320, 0.83 Å | ✗ | partial |
| `cryosparc_P7_J264_passthrough_particles_all_classes_ctf.cs` | 301,770 | ✗ | ✗ | ✓ full `ctf/*` |
| `cryosparc_P7_J264_consensus_pose.cs` | 301,770 | ✗ | `alignments3D/*` single-class (built locally, Step 1) | ✗ |

Per-class files (9 protein classes) in `data/J264_classes/`:
`cryosparc_P7_J264_class_06..14_00062_particles.cs` (+ `_volume_sharp.mrc`) and
`cryosparc_P7_J264_passthrough_particles_class_6..14.cs`.

**Inputs already generated locally (checked in):**
- `results_cryodrgn/J264_real/inputs/poses.pkl` — rot (301770,3,3) + trans (301770,2)
- `results_cryodrgn/J264_real/inputs/ctf.pkl` — (301770,9)

**One thing still needed on the cluster before `downsample`:**
The raw particle pixels. `blob/path` → `J226/reconstructed/<uid>_particles.mrc`
on the CryoSPARC server. Rsync/copy the `J226/reconstructed/` directory to the
cluster and set `$IMAGES_DIR` to the directory that makes
`J226/reconstructed/…mrc` resolve (the directory *above* `J226/`).

In the commands below:
- `$IMAGES_DIR` = parent of `J226/` on the cluster
- All commands run from the repo root: `cd /home/mae2183/cryoem-classification`

---

## Step 1 — Parse poses and CTF  ✅ DONE LOCALLY

Both files are already generated and checked in:

| file | shape | note |
|---|---|---|
| `results_cryodrgn/J264_real/inputs/poses.pkl` | rot (301770,3,3) float32 + trans (301770,2) float32 | best-class argmax pose extracted from `alignments3D_multi/pose` |
| `results_cryodrgn/J264_real/inputs/ctf.pkl` | (301770,9) float32 | parsed from `data/J264/…_ctf.cs` via `parse_ctf_csparc` |

**How poses were derived.** `cryodrgn parse_pose_csparc` expects a single-class
`.cs` (reads `alignments3D/pose`). J264 only has `alignments3D_multi/pose` (shape
301770×15×3). A reusable script rebuilds a single-class `.cs` by taking, for each
particle, the pose of its most-probable class (argmax of
`alignments3D_multi/class_posterior`), then cryoDRGN's own parser does the
rotation-transpose + shift conversion:

```bash
# 1a. build the consensus single-class .cs (local, metadata only)
python scripts/cryodrgn/build_consensus_pose_cs.py \
    --multi data/J264/cryosparc_P7_J264_00062_particles_alignments3D_multi.cs \
    -o data/J264/cryosparc_P7_J264_consensus_pose.cs

# 1b. parse poses (local)
cryodrgn parse_pose_csparc \
    data/J264/cryosparc_P7_J264_consensus_pose.cs \
    -D 320 \
    -o results_cryodrgn/J264_real/inputs/poses.pkl

# 1c. parse CTF (local)
cryodrgn parse_ctf_csparc \
    data/J264/cryosparc_P7_J264_passthrough_particles_all_classes_ctf.cs \
    -D 320 --Apix 0.83 \
    -o results_cryodrgn/J264_real/inputs/ctf.pkl
```

> Caveat (same as J64): the ideal pose source is a C1 consensus refinement. The
> argmax-class pose is the established workable substitute — acceptable because
> the 9 classes sit close together in a continuous landscape. The assigned-class
> fractions are ~[6:0.22, 7:0.19, 8:0.15, 9:0.17, 10:0.14, 11:0.06, 12:0.03,
> 13:0.03, 14:0.01]; the 6 dummy classes hold <1% combined.

---

## Step 2 — Downsample images (cluster)

Use the passthrough blob `.cs` (it has `blob/path`) with `--datadir $IMAGES_DIR`:

```bash
# Pilot resolution D=128 (~2.1 Å/pix) — paper Fig. 5 filtering stage
cryodrgn downsample \
    data/J264/cryosparc_P7_J264_passthrough_particles_all_classes_blob.cs \
    -D 128 \
    --datadir $IMAGES_DIR \
    -o results_cryodrgn/J264_real/inputs/particles.128.mrcs

# Final high-res D=256 (~1.04 Å/pix) — paper Fig. 5 high-res stage
# (run after Step 4 once you know filtering will proceed)
cryodrgn downsample \
    data/J264/cryosparc_P7_J264_passthrough_particles_all_classes_blob.cs \
    -D 256 \
    --datadir $IMAGES_DIR \
    -o results_cryodrgn/J264_real/inputs/particles.256.mrcs
```

---

## Step 3 — Pilot training: 1D + 10D latent at D=128 (GPU)

Architecture: encoder = decoder = **256 × 3**, **50 epochs** (paper Fig. 5;
matches the J1442/J1497/J64 pilots).

```bash
# --- 1-D latent pilot: Fig. 5a z-histogram; junk = the z <= -1 tail ---
cryodrgn train_vae \
    results_cryodrgn/J264_real/inputs/particles.128.mrcs \
    --poses results_cryodrgn/J264_real/inputs/poses.pkl \
    --ctf   results_cryodrgn/J264_real/inputs/ctf.pkl \
    --zdim 1 \
    -n 50 \
    --enc-dim 256 --enc-layers 3 \
    --dec-dim 256 --dec-layers 3 \
    -o results_cryodrgn/J264_real/pilot_z1

# --- 10-D latent pilot: UMAP; junk = outlier 5-component GMM cluster ---
cryodrgn train_vae \
    results_cryodrgn/J264_real/inputs/particles.128.mrcs \
    --poses results_cryodrgn/J264_real/inputs/poses.pkl \
    --ctf   results_cryodrgn/J264_real/inputs/ctf.pkl \
    --zdim 10 \
    -n 50 \
    --enc-dim 256 --enc-layers 3 \
    --dec-dim 256 --dec-layers 3 \
    -o results_cryodrgn/J264_real/pilot_z10

# Built-in analysis for the 10-D pilot (UMAP, PCA, kmeans20 volumes)
cryodrgn analyze results_cryodrgn/J264_real/pilot_z10 49
```

Outputs in `results_cryodrgn/J264_real/pilot_z10/`: `z.49.pkl` (301,770 × 10),
`weights.49.pkl` + `config.yaml`, `analyze.49/` (UMAP/PCA plots, `kmeans20/`,
notebooks).

---

## Step 4 — Inspect and filter junk

**4a. Visualize the junk filter (local, on the synced z.pkl):**

```bash
.\cryodrgn-py310\Scripts\python.exe scripts/cryodrgn/cryodrgn_paper_figures.py \
    --z results_cryodrgn/J264_real/pilot_z10/z.49.pkl \
    -o results_cryodrgn/J264_real/pilot_z10/paper_figures
```

Writes `fig_junk_filter.png` (5-component GMM + `|z|`, flags the outlier cluster)
and `fig_pc1_histogram.png` (check for the z ≤ −1 tail).

**4b. Build the kept-particle index (cluster, in `cryoDRGN_filtering.ipynb`):**

```python
import pickle, numpy as np
from sklearn.mixture import GaussianMixture
N = 301770
# 1-D criterion
z1 = np.array(pickle.load(open('results_cryodrgn/J264_real/pilot_z1/z.49.pkl','rb')))
kept_1d = np.where(z1.ravel() > -1.0)[0]
# 10-D criterion
z10 = np.array(pickle.load(open('results_cryodrgn/J264_real/pilot_z10/z.49.pkl','rb')))
gmm = GaussianMixture(5, covariance_type='full', n_init=3, random_state=0).fit(z10)
labels = gmm.predict(z10); zmag = np.linalg.norm(z10, axis=1)
junk_comp = np.argmax([zmag[labels==c].mean() for c in range(5)])
kept_10d = np.where(labels != junk_comp)[0]
# intersection
kept = np.intersect1d(kept_1d, kept_10d)
print(f"Kept {len(kept):,} / {N} = {len(kept)/N:.1%}")
pickle.dump(kept, open('results_cryodrgn/J264_real/inputs/ind_keep.pkl','wb'))
```

Optional QC: 2D-classify the *removed* particles in CryoSPARC — expect ice,
edge hits, junk.

---

## Step 5 — Final high-resolution 10-D training (GPU)

Train on **kept particles only** at D=256, 1024×3, 100 epochs (paper Fig. 5e–g).

```bash
cryodrgn train_vae \
    results_cryodrgn/J264_real/inputs/particles.256.mrcs \
    --poses results_cryodrgn/J264_real/inputs/poses.pkl \
    --ctf   results_cryodrgn/J264_real/inputs/ctf.pkl \
    --ind   results_cryodrgn/J264_real/inputs/ind_keep.pkl \
    --zdim 10 \
    -n 100 \
    --enc-dim 1024 --enc-layers 3 \
    --dec-dim 1024 --dec-layers 3 \
    -o results_cryodrgn/J264_real/train_final

# Built-in analysis
cryodrgn analyze results_cryodrgn/J264_real/train_final 99
```

Outputs in `results_cryodrgn/J264_real/train_final/`: `z.99.pkl`, `analyze.99/`.

---

## Step 6 — How many metastable states? + representative maps

**6a. Re-run analysis with candidate k (start k=3, also try 2 and 5):**

```bash
cryodrgn analyze results_cryodrgn/J264_real/train_final 99 --ksample 3
```

**6b. All paper latent-space figures (local, after syncing z.pkl):**

```bash
.\cryodrgn-py310\Scripts\python.exe scripts/cryodrgn/cryodrgn_paper_figures.py \
    --z   results_cryodrgn/J264_real/train_final/z.99.pkl \
    --cs  data/J264/cryosparc_P7_J264_00062_particles_alignments3D_multi.cs \
    --passthrough-cs data/J264/cryosparc_P7_J264_passthrough_particles_all_classes_blob.cs \
    --protein-idx 6 7 8 9 10 11 12 13 14 --n-dummies 6 \
    --run-log results_cryodrgn/J264_real/train_final/run.log \
    -o    results_cryodrgn/J264_real/train_final/paper_figures
```

**6c. Render representative volumes at k-means centres (cluster):**

```bash
cryodrgn eval_vol \
    results_cryodrgn/J264_real/train_final/weights.99.pkl \
    --config results_cryodrgn/J264_real/train_final/config.yaml \
    --zfile  results_cryodrgn/J264_real/train_final/analyze.99/kmeans3/centers.txt \
    -o results_cryodrgn/J264_real/train_final/vols_kmeans3
```

---

## Step 7 — Free energy + basins → which hetero-refine to run next (local)

This is the payoff step. After syncing `z.99.pkl`:

```bash
# 7a. Free-energy landscape F(PC1) = -log p(PC1): count the metastable minima
.\cryodrgn-py310\Scripts\python.exe scripts/cryodrgn/cryodrgn_free_energy.py \
    --dataset "J264:results_cryodrgn/J264_real/train_final/z.99.pkl:data/J264/cryosparc_P7_J264_passthrough_particles_all_classes_blob.cs:data/J264/cryosparc_P7_J264_00062_particles_alignments3D_multi.cs" \
    -o results_cryodrgn/J264_real/free_energy

# 7b. Basin occupancy: which of the 9 classes share a basin (+ export per-basin particles)
.\cryodrgn-py310\Scripts\python.exe scripts/cryodrgn/cryodrgn_basin_occupancy.py \
    --dataset "J264:results_cryodrgn/J264_real/train_final/z.99.pkl:data/J264/cryosparc_P7_J264_passthrough_particles_all_classes_blob.cs:data/J264/cryosparc_P7_J264_00062_particles_alignments3D_multi.cs:6,7,8,9,10,11,12,13,14" \
    --n-dummies 6 --barrier-kt 0.5 --export-basins \
    -o results_cryodrgn/J264_real/basin_occupancy

# 7c. Latent GMM confusion vs CryoSPARC (per-particle JS divergence)
.\cryodrgn-py310\Scripts\python.exe scripts/cryodrgn/cryodrgn_latent_gmm.py \
    --z results_cryodrgn/J264_real/train_final/z.99.pkl \
    --passthrough-cs data/J264/cryosparc_P7_J264_passthrough_particles_all_classes_blob.cs \
    --cs data/J264/cryosparc_P7_J264_00062_particles_alignments3D_multi.cs \
    --n-dummies 6 -k 3 \
    --outdir results_cryodrgn/J264_real/latent_gmm
```

**Read-out.** The basin-occupancy matrix (rows = 9 CryoSPARC classes, cols =
cryoDRGN basins) tells you the follow-up hetero-refine plan directly:
- classes that map to **the same basin** = one energetic state → merge their
  particles (the exported `basin_particles/*.cs` are CryoSPARC-importable) and,
  if that basin is structurally split, run a **focused K = (#classes in basin)**
  hetero-refine to recover substates;
- classes that each own **their own basin** = genuinely distinct states → K=1,
  no further splitting needed.

If cryoDRGN resolves **< 9** basins, that is the headline result: the 9 classes
over-fragment a smaller set of energetic states, and Step 7b names exactly which
to merge.

---

## Baseline already computed (no GPU): CryoSPARC 9-class assignment confusion

Run locally before any training, to characterise how confident the *existing*
9-class labels are (a baseline the cryoDRGN basins are compared against):

```bash
.\cryodrgn-py310\Scripts\python.exe run_pipeline.py \
    --cs data/J264/cryosparc_P7_J264_00062_particles_alignments3D_multi.cs \
    --n-dummies 6 --transform alr --covariance full \
    --mc-samples 50000 --n-boot 30 --reps 0 1 2 3 \
    --outdir results_J264
```

Outputs in `results_J264/` (`confusion/`, `populations/`, `gmm/`, `exports/`).
Finding: the 9-class posteriors are **highly confident** — soft-posterior
confusion diagonal ≈ 0.99, Monte-Carlo / analytical diagonal = 1.0, max
off-diagonal Bhattacharyya overlap ≈ 0. I.e. CryoSPARC assigns particles to the
9 classes almost deterministically, so *label* uncertainty is low; whether those
9 labels correspond to 9 *distinct energetic states* is the question cryoDRGN
(Step 7) answers.

---

## Paper figures produced (non-map panels)

| Script output | Paper panel | What it shows |
|---|---|---|
| `fig_pca_density.png` | Fig. 4c,e | PCA with `PC1 (EV, x.xx)` axis labels |
| `fig_umap_density.png` | Fig. 5b | UMAP (k=15, min_dist=0.1) density map |
| `fig_pc1_histogram.png` | Fig. 3e–h / 5a | PC1 histogram; modes = candidate states |
| `fig_latent_by_class.png` | Fig. 5c,d | Latent coloured by CryoSPARC class + class-mean stars |
| `fig_junk_filter.png` | Fig. 5c | 5-comp GMM + `\|z\|`; junk cluster highlighted |
| `fig_training_curve.png` | Fig. 2c,d | Training loss vs epoch from `run.log` |
| `kmeans_centers.txt` | Fig. 5f,g | On-data k-means centres for `eval_vol --zfile` |

---

## Command cheat-sheet (order of operations)

1. **Local, done:** poses.pkl + ctf.pkl (Step 1) + CryoSPARC-confusion baseline.
2. **Cluster:** copy `J226/reconstructed/` → set `$IMAGES_DIR` → `downsample` D=128 (Step 2).
3. **Cluster GPU:** pilot z1 + z10 @ D=128 (Step 3) → `cryodrgn analyze`.
4. **Pull**, inspect junk locally (Step 4a) → build `ind_keep.pkl` (Step 4b).
5. **Cluster:** `downsample` D=256 → **GPU** final z10 @ D=256 (Step 5) → `analyze --ksample`.
6. **Pull** `z.99.pkl` → run Step 6b + all of Step 7 locally → read basin matrix → plan focused hetero-refine.
