# John's meeting follow-ups — what was done (cryoDRGN / J1442)

This note maps each item from the meeting notes to the workflow output produced
locally. Everything here uses the converged J1442 cryoDRGN run
(`train_z10`, zdim=10, 100 epochs, z.100.pkl). The actual ab-initio/NU/refinement
jobs run in CryoSPARC on the GPU server — all the **particle sets and metrics**
needed to launch them are prepared here.

---

## 1. "Run NU on cryoDRGN particles" + "Run NU on low uncertainty (maybe a hard-assignments-only one?)"
`results_cryodrgn/J1442_real/cryodrgn_subsets/` — four confidence tiers per class
(import each `.cs` + run Ab-initio → NU):

| class | full | hard (argmax==CryoSPARC) | hcperm (permissive) | hc (strict) |
|---|---|---|---|---|
| P6 | 92,453 | 75,560 | 56,591 | 26,203 |
| P7 | 54,371 | 42,915 | 34,930 | 15,573 |
| P8 | 83,572 | 68,956 | 54,434 | 31,211 |

- **full** = run NU on all cryoDRGN-assigned particles.
- **hard** = hard assignments only (the "hard assignments only" run John asked about).
- **hcperm** = NEW permissive low-uncertainty set (JS < 66th pct). This is the
  "another confidence set because the current one cuts out too many" — it keeps
  roughly **2× more particles** than the strict `hc` set while still removing the
  most ambiguous particles.
- **hc** = strict low-uncertainty (the original set; smallest/purest).

## 2. Ab-initio (init 12 Å, final 4 Å) on a crude PC1 division + 3-component GMM + most-confident
`results_cryodrgn/J1442_real/pc1_classify/` — the **3 ab-initio→NU runs**:

- **RUN 1 — crude PC1 division** (equal-population tertiles along PC1):
  `pc1_crude_P{6,7,8}.cs` ≈ 76,799 each.
- **RUN 2 — 3-component GMM on PC1, confident only** (resp ≥ 0.8, 95.5% kept):
  `pc1_gmm_conf_P{6,7,8}.cs` = 91,885 / 43,040 / 85,101
  (`pc1_gmm_P*.cs` = all hard-assigned, no confidence cut).
- **RUN 3 — full latent-space GMM**: the `cryodrgn_subsets/cryodrgn_class_P*.cs`
  above (already the full multidimensional latent partition).

Use Ab-initio initial resolution 12 Å / final 4 Å, then NU on each.

## 3. "Particle overlap there too" + "a confusion metric for cryoDRGN like CryoSPARC's witness test"
`pc1_classify/overlap_metrics.json` + `confusion_*.png`:

- Pairwise **ARI** (1=identical, 0=chance):
  - PC1-GMM vs full latent-GMM = **0.927** → PC1 alone reproduces nearly the
    whole latent partition (the latent is effectively 1-D).
  - cryoDRGN vs CryoSPARC: crude **0.452**, PC1-GMM **0.519**, latent-GMM **0.543**
    → cryoDRGN's split agrees with the CryoSPARC hetero classes only moderately
    (~½), i.e. it is **not** the same partition.
- **Row-normalised confusion vs CryoSPARC** (the cryoDRGN analogue of CryoSPARC's
  class-population witness): PC1-crude diagonal [0.77, 0.71, 0.82] — P7 (middle PC1
  bin) is the most confused, P6/P8 cleaner.
- **Self soft-posterior witness confusion** of the PC1-GMM
  (`confusion_pc1gmm_witness.png`) — same idea as the hetero-refinement witness:
  how much each cryoDRGN class leaks into the others under its own soft weights.

## 4. "PC1 too confident / suspect overfitting / make sure the model isn't overfitting"
`results_cryodrgn/J1442_real/overfit_check/` — decisive, nuanced answer:

- **The cryoDRGN model is NOT overfitting**:
  - PC1 is **free of imaging confounds** (max |Pearson r| = 0.02 with defocus /
    astigmatism / phase / per-particle scale / pose / shift; multivariate
    R² = 0.00) → PC1 encodes **structure, not CTF/pose artifacts**.
  - PC1 is **stable across epochs** (locks to its final direction by epoch ~73,
    |corr|→1.0) → converged coordinate, not late-training memorisation.
  - Train loss converged (tail-20 change 0.01%).
- **But the discrete classification IS over-confident** (this is what John saw):
  the K=3 latent-GMM assigns **0.997** mean confidence while its components are
  only **1.90 SD** apart — for Gaussians that close the expected confidence is
  ~0.83, so there is a **+0.17 over-confidence gap**. The GMM *manufactures*
  confidence on a continuous cloud; it is not evidence of three separated wells.
- **Caveat**: cryoDRGN `train_vae` holds out no validation set, so a true
  train/val gap is unavailable — the three checks above are the substitute.

## 5. "See if cryoDRGN uncovered 5 classes in the 5-class set" — BLOCKED (needs GPU)
There is currently **no cryoDRGN run on the 5-class set** (J1497 / J1112). The only
trained cryoDRGN model in the workspace is the 3-class J1442 run. Training cryoDRGN
on the 5-class stack requires:
  1. the **raw particle .mrcs** (not present locally — the hard blocker noted in
     repo memory; all local `.cs` are metadata-only), and
  2. a **GPU** (local torch is CPU-only; 230k×320² is infeasible on CPU).

To do this: run cryoDRGN `train_vae` on the 5-class stack on the CryoSPARC/GPU
server (parse poses+CTF from the J1112/J1497 passthrough, downsample, train zdim≈10),
then re-run `cryodrgn_pc1_classify.py` and `cryodrgn_latent_gmm.py` with `-k 5`
to test whether the latent splits into 5 states. The analysis scripts are already
K-agnostic and will work as-is once a 5-class `z.N.pkl` exists.

---

### Scripts
- `scripts/cryodrgn/cryodrgn_pc1_classify.py` (new) — PC1 crude + GMM classification,
  overlap/confusion, `.cs` exports.
- `scripts/cryodrgn/cryodrgn_overfit_check.py` (new) — overfitting / over-confidence
  diagnostics.
- `scripts/cryodrgn/export_cryodrgn_subsets.py` (updated) — added `hard` and
  `hcperm` confidence tiers.
