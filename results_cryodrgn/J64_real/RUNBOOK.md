# J64 cryoDRGN runbook — following Zhong et al. (Nat. Methods 2021)

Follows the **Fig. 5 / EMPIAR-10076 assembly-state** workflow from Zhong et al.
(*Nature Methods* **18**, 176–185, 2021): pilot low-res runs to filter junk,
then a high-res final run to resolve states.

- **Cluster:** `mae2183@hudson`, repo root `/home/mae2183/cryoem-classification/`,
  conda env `cryodrgn` (`/home/mae2183/miniconda3/envs/cryodrgn/bin/cryodrgn`)
- **Local workspace:** `C:\Users\maemm\OneDrive\Desktop\CryoEM\` (this repo,
  synced via git push/pull)
- **J64 particles:** 702,919 particles, box 320, 0.83 Å/pix

---

## File inventory — what you have

All J64 `.cs` files are in `data/J64/`:

| file | rows | has blob | has 3D poses | has CTF |
|---|---|---|---|---|
| `cryosparc_P25_J64_00102_particles.cs` | 702,919 | ✗ | `alignments3D_multi/*` (17-class, used to derive poses — see Step 1) | ✗ |
| `cryosparc_P25_J64_passthrough_particles_all_classes_blob.cs` | 702,919 | ✓ `J21/extract/…_particles.mrc`, D=320, 0.83 Å | ✗ | ✓ |

**Inputs already generated locally (checked in):**
- `results_cryodrgn/J64_real/inputs/poses.pkl` — rot (702919,3,3) + trans (702919,2)
- `results_cryodrgn/J64_real/inputs/ctf.pkl` — (702919,9)

**One thing still needed on the cluster before `downsample`:**
The raw particle pixels. `blob/path` → `J21/extract/<uid>_…_particles.mrc` on the
CryoSPARC server. Rsync or copy the `J21/extract/` directory to the cluster and set
`$IMAGES_DIR` to the directory that makes `J21/extract/…mrc` resolve (the directory
*above* `J21/`).

In the commands below:
- `$IMAGES_DIR` = parent of `J21/` on the cluster
- All commands run from the repo root: `cd /home/mae2183/cryoem-classification`

---

## Step 1 — Parse poses and CTF  ✅ DONE LOCALLY

Both files are already generated and checked in:

| file | shape | note |
|---|---|---|
| `results_cryodrgn/J64_real/inputs/poses.pkl` | rot (702919,3,3) float32 + trans (702919,2) float32 | best-class argmax pose extracted from `alignments3D_multi/pose` |
| `results_cryodrgn/J64_real/inputs/ctf.pkl` | (702919,9) float32 | parsed from `data/J64/…_blob.cs` via `parse_ctf_csparc` |

**How poses were derived:** `cryodrgn parse_pose_csparc` is designed for a single-class
`.cs` (reads `alignments3D/pose`). J64 only has `alignments3D_multi/pose` (shape
702919×17×3). The equivalent was computed directly:

```python
best  = np.argmax(class_posterior, axis=1)          # argmax over 17 classes
ax_angle = pose_multi[np.arange(N), best, :]        # best-class axis-angle (N,3)
rot   = expmap(torch.tensor(ax_angle)).numpy()       # Rodrigues -> 3x3
rot   = np.array([r.T for r in rot], dtype=np.float32)  # transpose (cryodrgn convention)
trans = (shift_multi[np.arange(N), best, :] / psize).astype(np.float32)  # A -> pix
```

**CTF command that was run:**
```bash
cryodrgn parse_ctf_csparc \
    data/J64/cryosparc_P25_J64_passthrough_particles_all_classes_blob.cs \
    -D 320 --Apix 0.83 \
    -o results_cryodrgn/J64_real/inputs/ctf.pkl
```

---

## Step 2 — Downsample images

Use the passthrough blob `.cs` (it has `blob/path`) with `--datadir $IMAGES_DIR`:

```bash
# Pilot resolution D=128 (~3.3 Å/pix) — paper Fig. 5 filtering stage
cryodrgn downsample \
    data/J64/cryosparc_P25_J64_passthrough_particles_all_classes_blob.cs \
    -D 128 \
    --datadir $IMAGES_DIR \
    -o results_cryodrgn/J64_real/inputs/particles.128.mrcs

# Final high-res D=256 (~1.7 Å/pix) — paper Fig. 5 high-res stage
# (can run this after step 4 once you know filtering will proceed)
cryodrgn downsample \
    data/J64/cryosparc_P25_J64_passthrough_particles_all_classes_blob.cs \
    -D 256 \
    --datadir $IMAGES_DIR \
    -o results_cryodrgn/J64_real/inputs/particles.256.mrcs
```

---

## Step 3 — Pilot training: 1D + 10D latent at D=128

Run both pilots on a GPU node. Architecture: encoder = decoder = **256 × 3**,
**50 epochs** (paper Fig. 5). These match the J1442/J1497 pilot architecture
exactly (the same 256×3/50ep setting was used for the real J1442 run).

```bash
# --- 1-D latent pilot ---
# Used for: Fig. 5a-style z histogram; junk = the z <= -1 tail
cryodrgn train_vae \
    results_cryodrgn/J64_real/inputs/particles.128.mrcs \
    --poses results_cryodrgn/J64_real/inputs/poses.pkl \
    --ctf   results_cryodrgn/J64_real/inputs/ctf.pkl \
    --zdim 1 \
    -n 50 \
    --enc-dim 256 --enc-layers 3 \
    --dec-dim 256 --dec-layers 3 \
    -o results_cryodrgn/J64_real/pilot_z1

# --- 10-D latent pilot ---
# Used for: UMAP embedding; junk = outlier 5-component GMM cluster
cryodrgn train_vae \
    results_cryodrgn/J64_real/inputs/particles.128.mrcs \
    --poses results_cryodrgn/J64_real/inputs/poses.pkl \
    --ctf   results_cryodrgn/J64_real/inputs/ctf.pkl \
    --zdim 10 \
    -n 50 \
    --enc-dim 256 --enc-layers 3 \
    --dec-dim 256 --dec-layers 3 \
    -o results_cryodrgn/J64_real/pilot_z10

# Built-in analysis for the 10-D pilot (UMAP, PCA, kmeans20 volumes)
cryodrgn analyze results_cryodrgn/J64_real/pilot_z10 49
```

Output files in `results_cryodrgn/J64_real/pilot_z10/`:
- `z.49.pkl` — latent encodings (702,919 × 10)
- `weights.49.pkl` + `config.yaml` — model checkpoint
- `analyze.49/` — UMAP, PCA plots, `kmeans20/centers.txt`, Jupyter notebooks

---

## Step 4 — Inspect and filter junk

**4a. Visualize the junk filter (run locally on the already-synced z.pkl):**

```bash
python scripts/cryodrgn/cryodrgn_paper_figures.py \
    --z results_cryodrgn/J64_real/pilot_z10/z.49.pkl \
    -o results_cryodrgn/J64_real/pilot_z10/paper_figures
```

This writes `fig_junk_filter.png` (5-component GMM + `|z|` magnitude, flags
the outlier cluster) and `fig_pc1_histogram.png` (check for the z ≤ −1 tail).

**4b. Build the kept-particle index (run on the cluster):**

Open the `cryoDRGN_filtering.ipynb` notebook auto-generated in
`results_cryodrgn/J64_real/pilot_z10/analyze.49/` and apply both criteria:

1. **1-D criterion** — load `results_cryodrgn/J64_real/pilot_z1/z.49.pkl` and
   keep indices where `z > -1.0`:
   ```python
   import pickle, numpy as np
   z1 = np.array(pickle.load(open('results_cryodrgn/J64_real/pilot_z1/z.49.pkl','rb')))
   kept_1d = np.where(z1.ravel() > -1.0)[0]
   ```
2. **10-D criterion** — fit a 5-component GMM, identify the outlier component by
   largest mean `|z|`, keep all others:
   ```python
   from sklearn.mixture import GaussianMixture
   z10 = np.array(pickle.load(open('results_cryodrgn/J64_real/pilot_z10/z.49.pkl','rb')))
   gmm = GaussianMixture(5, covariance_type='full', n_init=3, random_state=0).fit(z10)
   labels = gmm.predict(z10)
   zmag = np.linalg.norm(z10, axis=1)
   junk_comp = np.argmax([zmag[labels==c].mean() for c in range(5)])
   kept_10d = np.where(labels != junk_comp)[0]
   ```
3. **Intersection** and save:
   ```python
   kept = np.intersect1d(kept_1d, kept_10d)
   print(f"Kept {len(kept):,} / 702919 = {len(kept)/702919:.1%}")
   pickle.dump(kept, open('results_cryodrgn/J64_real/inputs/ind_keep.pkl','wb'))
   ```

Optional QC: 2D-classify the *removed* particles in CryoSPARC — they should be
ice, edge hits, and other junk.

---

## Step 5 — Final high-resolution 10-D training

Train on the **kept particles only** at D=256 with the larger 1024×3 architecture
(paper Fig. 5e–g). Matches J1497 in training length (100 epochs).

```bash
cryodrgn train_vae \
    results_cryodrgn/J64_real/inputs/particles.256.mrcs \
    --poses results_cryodrgn/J64_real/inputs/poses.pkl \
    --ctf   results_cryodrgn/J64_real/inputs/ctf.pkl \
    --ind   results_cryodrgn/J64_real/inputs/ind_keep.pkl \
    --zdim 10 \
    -n 100 \
    --enc-dim 1024 --enc-layers 3 \
    --dec-dim 1024 --dec-layers 3 \
    -o results_cryodrgn/J64_real/train_final

# Built-in analysis
cryodrgn analyze results_cryodrgn/J64_real/train_final 99
```

Output files in `results_cryodrgn/J64_real/train_final/`:
- `z.99.pkl` — final latent encodings
- `analyze.99/` — UMAP, PCA, kmeans20, `cryoDRGN_filtering.ipynb`

---

## Step 6 — Classify into 3 or 5 states + generate representative maps

**6a. Re-run analysis with k=5 cluster centres:**

```bash
cryodrgn analyze results_cryodrgn/J64_real/train_final 99 --ksample 5
```

This writes `analyze.99/kmeans5/centers.txt` and renders 5 volumes at the
on-data cluster centres.

**6b. Generate all paper latent-space figures (run locally after syncing):**

```bash
python scripts/cryodrgn/cryodrgn_paper_figures.py \
    --z   results_cryodrgn/J64_real/train_final/z.99.pkl \
    --cs  data/J64/cryosparc_P25_J64_00102_particles.cs \
    --passthrough-cs data/J64/cryosparc_P25_J64_passthrough_particles_all_classes_blob.cs \
    --run-log results_cryodrgn/J64_real/train_final/run.log \
    -o    results_cryodrgn/J64_real/train_final/paper_figures
```

**6c. Render representative volumes at k-means centres:**

```bash
# On the cluster — uses the centres from step 6a
cryodrgn eval_vol \
    results_cryodrgn/J64_real/train_final/weights.99.pkl \
    --config results_cryodrgn/J64_real/train_final/config.yaml \
    --zfile  results_cryodrgn/J64_real/train_final/analyze.99/kmeans5/centers.txt \
    -o results_cryodrgn/J64_real/train_final/vols_kmeans5
```

**6d. Run the full latent-space analysis pipeline (same scripts as J1442/J1497):**

```bash
# Latent GMM + population CIs + per-particle JS divergence vs CryoSPARC
python scripts/cryodrgn/cryodrgn_latent_gmm.py \
    --z results_cryodrgn/J64_real/train_final/z.99.pkl \
    --passthrough-cs data/J64/cryosparc_P25_J64_passthrough_particles_all_classes_blob.cs \
    --cs data/J64/cryosparc_P25_J64_00102_particles.cs \
    --n-dummies 0 \
    -k 3 \
    --outdir results_cryodrgn/J64_real/latent_gmm

# Free-energy landscape F(PC1) = -log p(PC1)
python scripts/cryodrgn/cryodrgn_free_energy.py \
    --dataset "J64:results_cryodrgn/J64_real/train_final/z.99.pkl:data/J64/cryosparc_P25_J64_passthrough_particles_all_classes_blob.cs:data/J64/cryosparc_P25_J64_00102_particles.cs" \
    -o results_cryodrgn/J64_real/free_energy
```

---

## Paper figures produced (non-map panels)

| Script output | Paper panel | What it shows |
|---|---|---|
| `fig_pca_density.png` | Fig. 4c,e | PCA with `PC1 (EV, x.xx)` axis labels |
| `fig_umap_density.png` | Fig. 5b | UMAP (k=15, min_dist=0.1) density map |
| `fig_pc1_histogram.png` | Fig. 3e–h / 5a | PC1 histogram; modes = candidate states |
| `fig_latent_by_class.png` | Fig. 5c,d | Latent coloured by CryoSPARC class + on-data class mean stars |
| `fig_junk_filter.png` | Fig. 5c | 5-comp GMM + `\|z\|`; junk cluster highlighted |
| `fig_training_curve.png` | Fig. 2c,d | Training loss vs epoch from `run.log` |
| `kmeans_centers.txt` | Fig. 5f,g | On-data k-means centres for `eval_vol --zfile` |

---

## Is it worth the time?

**Yes — run it as a junk-filter and rare-state finder, not to force tidy classes.**
Prior J1442/J1497 analysis shows the latent is a continuous 1-D reaction
coordinate; discrete classes are arbitrary slices. J64 is the upstream, junkier
parent, which is exactly where cryoDRGN's Fig. 5 workflow pays off:
(1) **impurity removal** via latent-space outlier rejection (the main win), and
(2) **rare-state discovery** that discrete classification misses.
The D=128 pilot runs (step 3) are cheap on a GPU (~1–2 h for 700k particles);
only commit to the D=256 final run (step 5) if the pilots show structure beyond
the continuum.
