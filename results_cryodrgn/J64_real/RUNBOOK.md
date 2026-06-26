# J64 cryoDRGN runbook — following Zhong et al. (Nat. Methods 2021)

This reproduces the heterogeneity workflow from **"CryoDRGN: reconstruction of
heterogeneous cryo-EM structures using neural networks"** (Zhong, Bepler,
Berger & Davis, *Nature Methods* **18**, 176–185, 2021), specifically the
**Fig. 5 / EMPIAR-10076 assembly-state pipeline** — the direct analog of J64:
a large, junky multi-class set that is filtered in latent space and then
re-trained at high resolution to resolve discrete states.

- **Input set:** `data/J64/cryosparc_P25_J64_00102_particles.cs`
  (702,919 particles; the 17-class hetero-refinement input that precedes
  J1442/J1497; Apix = 0.83 Å).
- **Outputs:** `results_cryodrgn/J64_real/`

---

## 0. Prerequisite — what you have vs. what is still missing

Three J64 files are now in `data/J64/`:

| file | rows | provides | usable for |
|---|---|---|---|
| `cryosparc_P25_J64_00102_particles.cs` | 702,919 | `uid` + `alignments3D_multi/*` (17-class poses/posteriors) | the existing CryoSPARC classification only |
| `cryosparc_P25_J64_passthrough_particles_all_classes_blob.cs` | 702,919 | `blob/path,idx,shape,psize,sign` (image pointers, box **320**, **0.83 Å/pix**) + full `ctf/*` | **images + CTF** |
| `cryosparc_P25_J64_passthrough_particles_all_classes_ctf.cs`  | 702,919 | same `blob/*` + `ctf/*` | **CTF** |

> **What is present:** the raw-image POINTERS (`blob/path` → `J21/extract/<uid>_..._particles.mrc`, 320² @ 0.83 Å) and the full **CTF**. CTF and downsampling can be parsed straight from the passthrough `.cs` below.
>
> **What is still missing — 3D consensus poses.** The passthrough files carry only `alignments2D/*` (2D-class in-plane poses), and `..._particles.cs` carries only `alignments3D_multi/*` (the 17-class hetero poses). cryoDRGN `train_vae` needs ONE consensus 3D orientation per particle, which comes from a **C1 homogeneous / NU-refinement** of these same 702,919 particles. Export that consensus job's `*_particles.cs` (it will have `alignments3D/pose` + `alignments3D/shift`) and use it for `parse_pose_csparc` in step 1.
>
> **What is not in the workspace — the actual pixels.** `blob/path` points at `J21/extract/*.mrc` on the CryoSPARC box. Copy that `J21/extract/` tree to the cluster (or run cryoDRGN on the CryoSPARC server) and set `--datadir` to the directory that makes `J21/extract/...mrc` resolve.

In the commands below:
- `CONSENSUS_CS` = the C1 consensus refinement `*_particles.cs` (for 3D poses) — **the one file you still need to export**
- `IMAGES_DIR`   = base dir that resolves `J21/extract/<...>.mrc`
- box `D = 320`, `APIX = 0.83`

---

## 1. Parse poses + CTF

```bash
# 3D consensus poses (from the C1 refinement you export)
cryodrgn parse_pose_csparc $CONSENSUS_CS -D 320 \
    -o results_cryodrgn/J64_real/inputs/poses.pkl

# CTF — straight from the J64 passthrough already in data/J64/
cryodrgn parse_ctf_csparc \
    data/J64/cryosparc_P25_J64_passthrough_particles_all_classes_ctf.cs \
    -D 320 --Apix 0.83 \
    -o results_cryodrgn/J64_real/inputs/ctf.pkl
```

## 2. Downsample the images (paper: D=128 for pilots, D=256 for final)

The passthrough `_blob.cs` has the `blob/path` image pointers, so downsample
reads it directly (with `--datadir` pointing at the `J21/extract/` parent):

```bash
# pilot / filtering resolution (paper Fig. 5)
cryodrgn downsample \
    data/J64/cryosparc_P25_J64_passthrough_particles_all_classes_blob.cs \
    -D 128 --datadir $IMAGES_DIR \
    -o results_cryodrgn/J64_real/inputs/particles.128.mrcs

# high-resolution final-training resolution (paper Fig. 5)
cryodrgn downsample \
    data/J64/cryosparc_P25_J64_passthrough_particles_all_classes_blob.cs \
    -D 256 --datadir $IMAGES_DIR \
    -o results_cryodrgn/J64_real/inputs/particles.256.mrcs
```

---

## 3. Pilot/filtering runs — train BOTH a 1D and a 10D model (paper Fig. 5a,b)

Paper pilot architecture: encoder = decoder = **256 × 3**, **50 epochs**,
minibatch 8, Adam lr 1e-4 (cryoDRGN defaults). Run on a GPU node.

```bash
# 1-D latent (Fig. 5a histogram; junk = the z<=-1 tail)
cryodrgn train_vae results_cryodrgn/J64_real/inputs/particles.128.mrcs \
    --ctf   results_cryodrgn/J64_real/inputs/ctf.pkl \
    --poses results_cryodrgn/J64_real/inputs/poses.pkl \
    --zdim 1  -n 50 \
    --enc-dim 256 --enc-layers 3 --dec-dim 256 --dec-layers 3 \
    -o results_cryodrgn/J64_real/pilot_z1

# 10-D latent (Fig. 5b UMAP; junk = outlier GMM component)
cryodrgn train_vae results_cryodrgn/J64_real/inputs/particles.128.mrcs \
    --ctf   results_cryodrgn/J64_real/inputs/ctf.pkl \
    --poses results_cryodrgn/J64_real/inputs/poses.pkl \
    --zdim 10 -n 50 \
    --enc-dim 256 --enc-layers 3 --dec-dim 256 --dec-layers 3 \
    -o results_cryodrgn/J64_real/pilot_z10

cryodrgn analyze results_cryodrgn/J64_real/pilot_z10 49     # UMAP + PCA + kmeans
```

> The one-shot wrapper `scripts/cryodrgn/cryodrgn_run.py` assumes a single `.cs`
> that carries blob + 3D poses + CTF together. Here poses live in a *separate*
> consensus file from blob/CTF, so run the explicit `parse_* / downsample /
> train_vae` steps above (or pass the consensus `.cs` as `--particles` and the
> passthrough blob as `--datadir` source only if they share uid order).

---

## 4. Filter junk (paper Fig. 5c,d — the reason to run J64 at all)

The paper removes impurities in latent space, then keeps the **intersection** of
the 1D and 10D kept sets:

1. **1-D model:** remove particles with **z ≤ −1** (the low tail of the Fig. 5a
   histogram).
2. **10-D model:** fit a **five-component, full-covariance GMM** (scikit-learn)
   to the latent encodings and remove the **outlier component** (largest mean
   `|z|`).
3. Keep `kept_1D ∩ kept_10D`. (Optional QC: 2D-classify the *removed* particles —
   they should be ice/edge/junk.)

This repo automates the latent inspection + 5-component GMM + |z| outlier flag:

```bash
# writes fig_junk_filter.png and reports the junk component + its particle %
python scripts/cryodrgn/cryodrgn_paper_figures.py \
    --z results_cryodrgn/J64_real/pilot_z10/z.49.pkl \
    -o results_cryodrgn/J64_real/pilot_z10/paper_figures
```

Build the kept-index `.pkl` (intersection of the two criteria) with the cryoDRGN
filtering notebook (`cryodrgn_filtering.ipynb`, auto-written into each train dir)
or `cryodrgn_utils select_clusters`, and save it as
`results_cryodrgn/J64_real/inputs/ind_keep.pkl`.

---

## 5. Final high-resolution 10-D training (paper Fig. 5e–g)

Train a 10-D model on the kept particles at **D=256**, encoder = decoder =
**1024 × 3**, 50 epochs. (The paper trains on a random 90% of the kept set; add a
held-out 10% index if you want the same train/val split.)

```bash
cryodrgn train_vae results_cryodrgn/J64_real/inputs/particles.256.mrcs \
    --ctf   results_cryodrgn/J64_real/inputs/ctf.pkl \
    --poses results_cryodrgn/J64_real/inputs/poses.pkl \
    --ind   results_cryodrgn/J64_real/inputs/ind_keep.pkl \
    --zdim 10 -n 50 \
    --enc-dim 1024 --enc-layers 3 --dec-dim 1024 --dec-layers 3 \
    -o results_cryodrgn/J64_real/train_final

cryodrgn analyze results_cryodrgn/J64_real/train_final 49 --pc 2 --ksample 20
```

---

## 6. Classify into 3 or 5 states + representative maps (paper Fig. 5f,g)

The paper generates representative maps at **on-data k-means cluster centres**.
To force the 3 or 5 discrete classes you asked for, cluster the final latent at
k=3 or k=5 and render volumes at those centres:

```bash
# k=5 representative volumes at on-data cluster centres
cryodrgn analyze results_cryodrgn/J64_real/train_final 49 --pc 2 --ksample 5

# OR drive it from the paper-figures script, which also writes kmeans_centers.txt
# (--passthrough-cs = the stack you TRAINED on, so its uid order matches z)
python scripts/cryodrgn/cryodrgn_paper_figures.py \
    --z  results_cryodrgn/J64_real/train_final/z.49.pkl \
    --cs data/J64/cryosparc_P25_J64_00102_particles.cs \
    --passthrough-cs data/J64/cryosparc_P25_J64_passthrough_particles_all_classes_blob.cs \
    --run-log results_cryodrgn/J64_real/train_final/run.log \
    -o results_cryodrgn/J64_real/train_final/paper_figures

# render the 5 representative maps from the centres file
cryodrgn eval_vol results_cryodrgn/J64_real/train_final/weights.49.pkl \
    --config results_cryodrgn/J64_real/train_final/config.yaml \
    --zfile  results_cryodrgn/J64_real/train_final/paper_figures/kmeans_centers.txt \
    -o results_cryodrgn/J64_real/train_final/representative_maps
```

To compare directly against the existing CryoSPARC 17-class labelling, the paper
also reports the **on-data mean latent encoding per class**,
$\hat z_M = \frac{1}{|M|}\sum_{i\in M} z_i$; the figure script marks these as
stars when you pass `--cs/--passthrough-cs/--protein-idx`.

---

## Non-map figures reproduced by `cryodrgn_paper_figures.py`

Run after any `train_vae` to get the paper's latent-space panels (everything
except the 3D map renderings):

| Output PNG | Paper panel | What it shows |
|---|---|---|
| `fig_pca_density.png`    | Fig. 4c,e | PCA of the latent with `PC (EV, x.xx)` axis labels |
| `fig_umap_density.png`   | Fig. 5b   | UMAP (k=15, min_dist=0.1) density embedding |
| `fig_pc1_histogram.png`  | Fig. 3e–h / 5a | 1-D latent (PC1) histogram; modes = states |
| `fig_latent_by_class.png`| Fig. 5c,d | latent coloured by CryoSPARC class + on-data class means |
| `fig_junk_filter.png`    | Fig. 5c   | 5-component GMM + `\|z\|` magnitude; junk cluster flagged |
| `fig_training_curve.png` | Fig. 2c,d | training loss vs epoch (from `run.log`) |
| `kmeans_centers.txt`     | Fig. 5f,g | on-data k-means centres → `eval_vol --zfile` |

---

## Is it worth the time?

**Yes — but run it as a junk-filter / rare-state finder, not to force 3–5 tidy
classes.** Prior analysis in this workspace shows the J1442/J1497 latent is a
**continuous 1-D reaction coordinate**, so any "3 classes" or "5 classes" there
are arbitrary slices of a continuum. J64 is the *upstream, junkier* parent set,
which is exactly where cryoDRGN's two documented strengths (paper Fig. 5) pay
off: (1) **impurity removal** via latent-space outlier rejection, and (2)
**discovery of rare/under-represented states** (~1% populations) that 3D
classification misses. Expect the main wins to be a **cleaner particle stack**
(feed the kept indices back into CryoSPARC) and a **map of the true number of
states**, rather than a clean 3- or 5-way partition. The pilot D=128 runs (step
3) are cheap — do those first; only commit to the D=256 final run (step 5) if the
pilots show structure beyond the continuum.
