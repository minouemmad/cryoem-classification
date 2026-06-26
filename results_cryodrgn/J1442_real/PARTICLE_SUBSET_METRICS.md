# cryoDRGN J1442 — Particle Subsets, Classification Overlap, and Overfitting Diagnostics

All results use the **J1442 z10** run (`train_z10/`, zdim=10, 100 epochs, D=128).
Scripts: `scripts/cryodrgn/`. Output directories below are relative to
`results_cryodrgn/J1442_real/`.

---

## 1. Confidence-tiered particle subsets (`cryodrgn_subsets/`)

Four tiers of per-class `.cs` files, defined by how well the cryoDRGN latent-GMM
assignment and the CryoSPARC assignment agree:

| Tier | Definition | Import guidance |
|------|-----------|-----------------|
| `full` | all cryoDRGN latent-GMM hard-assigned particles | most particles; use for initial NU on cryoDRGN partition |
| `hard` | latent-GMM argmax == CryoSPARC argmax | reduces cross-method disagreement; ~82% of `full` |
| `hcperm` | `hard` AND JS(CryoSPARC, cryoDRGN) < 66th percentile | permissive low-uncertainty; ~2x more particles than `hc` |
| `hc` | `hard` AND JS(CryoSPARC, cryoDRGN) < 33rd percentile | strict low-uncertainty; purest signal, smallest set |

JS divergence threshold: `hcperm` = 0.381 nats; `hc` = 0.294 nats.

| class | full | hard | hcperm | hc |
|-------|------|------|--------|----|
| P6 | 92,453 | 75,560 | 56,591 | 26,203 |
| P7 | 54,371 | 42,915 | 34,930 | 15,573 |
| P8 | 83,572 | 68,956 | 54,434 | 31,211 |

All files contain the complete pose + CTF fields from the passthrough stack and
are directly importable into CryoSPARC. The sidecar `cryodrgn_posteriors.csv`
carries per-particle uid, cryoDRGN posterior, JS divergence, and agreement flag
for every particle.  Each file also has a `cryodrgn/class_posterior` field
injected (shape N x 3, float32, rows sum to 1.0).

**CryoSPARC workflow:** Import Particles -> Ab-initio (K=1) -> NU-Refinement.

---

## 2. PC1 classification and overlap metrics (`pc1_classify/`)

The 10-D latent was projected to PC1 (explains 15.5% of standardized variance)
and two alternative hard partitions were derived:

- **PC1 crude**: equal-population tertile cut along PC1 (~76,800 particles each).
- **PC1 GMM**: 3-component 1-D GMM fit along PC1; `conf` variant retains only
  particles with GMM responsibility >= 0.8 (95.5% of particles pass).

### Adjusted Rand Index between partitions (1 = identical, 0 = chance)

| partition A | partition B | ARI | AMI |
|-------------|-------------|-----|-----|
| CryoSPARC | PC1 crude | 0.452 | 0.404 |
| CryoSPARC | PC1 GMM | 0.519 | 0.448 |
| CryoSPARC | latent GMM (10-D) | 0.543 | 0.469 |
| PC1 crude | PC1 GMM | 0.699 | 0.711 |
| PC1 crude | latent GMM | 0.736 | 0.736 |
| **PC1 GMM** | **latent GMM** | **0.927** | **0.889** |

PC1-GMM vs full-latent-GMM ARI = 0.927: PC1 alone reproduces 93% of the full
10-D partition, confirming the latent is effectively 1-D (one dominant axis).
CryoSPARC agreement of 0.52-0.54 shows cryoDRGN and CryoSPARC are related but
distinct partitions (not the same discrete classification).

### Three particle set variants for ab-initio + NU-Refinement

1. `pc1_crude_P{6,7,8}.cs` -- equal-population PC1 tertile cut.
2. `pc1_gmm_conf_P{6,7,8}.cs` -- 1-D GMM (resp >= 0.8, 95.5% retained).
3. `cryodrgn_class_P{6,7,8}.cs` (in `cryodrgn_subsets/`) -- full 10-D GMM.

CryoSPARC: Ab-initio (initial resolution 12 A, final 4 A) then NU-Refinement.

---

## 3. Overfitting and over-confidence diagnostics (`overfit_check/`)

Four independent checks on the J1442 z10 run (zdim=10, 100 epochs):

### (1) Confidence-vs-separation paradox

The K=3 latent-GMM assigns **0.997 mean max-responsibility** while its components
are only **1.90 SD** apart.  For two Gaussians 1.90 SD apart the expected mean
max-probability (Monte Carlo, N=100,000) is ~0.83.  Observed gap: **+0.168** --
the GMM manufactures sharp assignments on a continuous distribution (over-confidence
artifact), not evidence of three well-separated wells.

### (2) Imaging-confound correlation

PC1 was correlated against all extractable per-particle imaging parameters
(defocus-U/V, astigmatism angle, phase shift, per-particle scale alpha,
pose rotation axes, shift magnitude):

- Max |Pearson r| = **0.02** (per-particle scale alpha)
- Multivariate R^2 (PC1 ~ all confounds): **0.00**

PC1 is not driven by CTF, defocus, or pose artifacts.

### (3) Epoch stability

PC1 correlation to the epoch-100 final PC1, sampled every 9 epochs:

| epoch | |r(PC1, final)| |
|-------|----------------|
| 1 | 0.057 |
| 10 | ~0.75 |
| 73 | >= 0.95 (lock-in) |
| 100 | 1.000 |

The coordinate converges monotonically and does not drift in late training.

### (4) Train-loss convergence

Generative loss from `run.log`: epoch 1 = 0.5624, epoch 10 = 0.5595,
epoch 100 = 0.5586.  Tail-20-epoch relative change: **0.01%** (converged).

### Verdict

The cryoDRGN model is **not overfitting**: PC1 is confound-free, stable,
and the loss is converged.  Over-confidence in the discrete GMM is an artifact
of fitting sharp components to a continuous distribution.
