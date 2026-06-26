# cryoDRGN J1497 (5-class) — Conformational Landscape Analysis

**Dataset:** J1497 (5-class CryoSPARC hetero-refinement, P6-P10, 6 dummy classes)
**Training:** zdim=10, 100 epochs, D=128; `results_cryodrgn/J1497_real/train/`

Both J1442 and J1497 CryoSPARC runs operated on the **identical** 230,396-particle
set (uid sets 100% identical). cryoDRGN was trained unsupervised on those images
for both jobs. The only difference is the number of CryoSPARC classes: 3 (J1442)
vs 5 (J1497).

---

## Key finding: the latent supports ~3 density regions, not 5

Forcing 5 GMM components degrades all separability metrics vs the 3-class case:

| metric | J1442 (3-class) | J1497 (5-class) |
|--------|-----------------|-----------------|
| min GMM component separation (SD) | 1.90 | **0.88** |
| silhouette at design K | 0.095 | **0.039** |
| PC1 kernel-density modes | 3 | 2-3 |
| BIC elbow at design K? | no (monotone) | no (monotone) |
| CryoSPARC <-> latent-GMM hard agreement | 0.814 | **0.561** |
| mean JS(CryoSPARC, latent-GMM) | 0.298 nats | **0.348 nats** |
| latent-GMM mean confidence | 0.997 | 0.902 |

Min separation 0.88 SD is far below the 2 SD threshold for discrete wells;
BIC decreases monotonically (no elbow): the 5 CryoSPARC classes are slicing
one continuous density.

---

## 1. Latent-space GMM (`latent_gmm/`)

5-component full-covariance GMM fit to standardized z.100 (230,396 x 10).
BIC sweep over K=1..10 showed no elbow at K=5 (BIC continues to decrease
to K=10), confirming no natural 5-cluster structure. See `latent_bic_sweep.png`.

Populations (500-replicate particle bootstrap, 95% CI):

| class | CryoSPARC soft | latent-GMM soft | latent-GMM corrected |
|-------|----------------|-----------------|----------------------|
| P6 | 0.199 | 0.380 [0.378, 0.382] | 0.226 [0.223, 0.229] |
| P7 | 0.203 | 0.225 [0.224, 0.227] | 0.172 [0.170, 0.175] |
| P8 | 0.200 | 0.148 [0.146, 0.149] | 0.313 [0.309, 0.316] |
| P9 | 0.199 | 0.208 [0.206, 0.209] | 0.046 [0.043, 0.049] |
| P10 | 0.198 | 0.039 [0.039, 0.040] | 0.243 [0.238, 0.247] |

The corrected populations for P9 and P10 (0.046, 0.243) are
inconsistent with the CryoSPARC soft fractions (~0.20 each), indicating
the latent-GMM cannot reliably recover these as distinct populations.

---

## 2. Landscape visualization (`landscape/`)

- PC1 explains 15.5%, PC2 11.2% of standardized latent variance (identical
  to J1442 — same images, same dominant axis).
- PC1 marginal density shows 2-3 peaks, not 5 (`latent_landscape.png`, panel D).
- K=10 fine-grain GMM piles redundant ellipses onto the same ~3 density peaks;
  no hidden 4th/5th distinct state visible (`latent_landscape_fine.png`).
- GMM peak z-vectors for eval_vol: `landscape/z_gmm_peaks.txt`

---

## 3. Supervised class recoverability (`crossjob_comparison/`)

Linear discriminant analysis (LDA), 5-fold stratified CV, latent z -> CryoSPARC
argmax class:

| run | balanced accuracy | chance | lift |
|-----|-------------------|--------|------|
| J1442 (3-class) | 0.799 | 0.333 | 2.40x |
| J1497 (5-class) | 0.485 | 0.200 | 2.43x |

Per-class LDA recall (J1497):

| class | recall | interpretation |
|-------|--------|----------------|
| P6 | 0.90 | recovered (same state as J1442 P6) |
| P7 | 0.64 | recovered (same state as J1442 P7) |
| P8 | 0.86 | recovered (same state as J1442 P8) |
| **P9** | **0.016** | 72% of P9 particles predict as P8 |
| **P10** | **0.010** | 67% of P10 particles predict as P6 |

P9 and P10 are structurally indistinguishable from P8 and P6, respectively.
The PC1 histogram confirms this: P9 peaks on P8's PC1 position, P10 on P6's.
See `crossjob_comparison/supervised_recall.png` and `crossjob_comparison/pc1_overlap.png`.

---

## 4. Alternative unsupervised classifiers (`crossjob_comparison/`)

ARI against CryoSPARC 5-class argmax:

| method | ARI | AMI |
|--------|-----|-----|
| MiniBatch KMeans (K=5) | 0.147 | 0.168 |
| Full-cov GMM (K=5) | 0.193 | 0.218 |
| **HDBSCAN (density clustering)** | **0.000** | **0.000** |

HDBSCAN (min_cluster_size=2000, subsampled 40k particles) found **0 dense
clusters** (100% noise fraction). This is the strongest evidence that the
density is continuous with no discrete high-density blobs.

---

## 5. Overfitting diagnostics (`overfit_check/`)

Same checks as J1442; all pass:
- Max |confound r| = 0.02 (df_angle_rad); multivariate R^2 = 0.00
- PC1 lock-in epoch: 82 (|corr| >= 0.95); epoch-100 corr = 1.000
- Final loss: 0.5588; tail-20 relative change: 0.01%
- Over-confidence: 0.902 mean confidence vs 0.671 expected at 0.88 SD
  (gap = 0.231; same artifact as J1442 but stronger)

---

## 6. Particle subsets (`cryodrgn_subsets/`)

Confidence-tiered `.cs` files (full/hard/hcperm/hc) with `cryodrgn/class_posterior`
field injected. Note the very small hc-tier counts for P9/P10.

| class | full | hard | hcperm | hc |
|-------|------|------|--------|----|
| P6 | 89,391 | 59,009 | 34,938 | 3,806 |
| P7 | 53,093 | 32,654 | 14,963 | 2,333 |
| P8 | 29,254 | 18,578 | 18,270 | 13,709 |
| P9 | 53,199 | 17,828 | 17,828 | 14,330 |
| P10 | 5,459 | 1,099 | 884 | 662 |

Overall hard-agreement: 56.1% (vs 73% for J1442), confirming the 5-class
partition is harder for cryoDRGN to reproduce than the 3-class one.

---

## Summary

The J1497 5-class CryoSPARC run did **not** resolve five conformations.
cryoDRGN recovered the same three states as J1442 (P6/P7/P8); P9 is a
sub-class of P8 and P10 a sub-class of P6. The continuous latent coordinate
is identical across both runs (CCA >= 0.97, PC1 r = 0.898). Recommended
representation: 3-state soft populations along the PC1 reaction coordinate.

To probe for hidden states: decode volumes along PC1 via `cryodrgn eval_vol`
with `landscape/z_gmm_peaks.txt`, then compare maps by local-resolution
difference (e.g., in ChimeraX). Alternatively, run a focused/masked 3D
classification on a region of interest rather than global hetero-refinement.

### Reproduce all analyses

```bash
# Latent-GMM
python scripts/cryodrgn/cryodrgn_latent_gmm.py \
  --z results_cryodrgn/J1497_real/train/z.100.pkl \
  --passthrough-cs data/gP25W6J1497_passthrough_particles_all_classes.cs \
  --cs data/cryosparc_P25_J1497_00000_particles.cs \
  --n-dummies 6 --protein-idx 6 7 8 9 10 -k 5 --k-max 10 \
  -o results_cryodrgn/J1497_real/latent_gmm

# Landscape
python scripts/cryodrgn/cryodrgn_landscape.py \
  --z results_cryodrgn/J1497_real/train/z.100.pkl \
  --passthrough-cs data/gP25W6J1497_passthrough_particles_all_classes.cs \
  --cs data/cryosparc_P25_J1497_00000_particles.cs \
  --n-dummies 6 --protein-idx 6 7 8 9 10 -k 5 --k-fine 10 \
  -o results_cryodrgn/J1497_real/landscape

# Cross-job comparison (J1442 reference, J1497 test)
python scripts/cryodrgn/cryodrgn_crossjob_compare.py \
  --z-ref results_cryodrgn/J1442_real/train_z10/z.100.pkl \
  --passthrough-ref data/cryosparc_P25_J1442_passthrough_particles_all_classes.cs \
  --cs-ref data/cryosparc_P25_J1442_00000_particles.cs \
  --protein-idx-ref 6 7 8 \
  --z-test results_cryodrgn/J1497_real/train/z.100.pkl \
  --passthrough-test data/gP25W6J1497_passthrough_particles_all_classes.cs \
  --cs-test data/cryosparc_P25_J1497_00000_particles.cs \
  --protein-idx-test 6 7 8 9 10 \
  --n-dummies 6 --label-ref "J1442 (3-class)" --label-test "J1497 (5-class)" \
  -o results_cryodrgn/J1497_real/crossjob_comparison
```
