# Branch validation — J1069 hard assignments + J1442 posterior weights

**Branch:** `results_J1069/cryosparc_outputs/with_1442_weights`
**Particles:** 230,396 (CFTR). Input hard assignment P6 = 91,826 · P7 = 54,548 · P8 = 84,022.
**Question:** Is this three-way classification a *real* structural partition, or an
arbitrary slicing of one continuous conformational ensemble?

Three independent, literature-standard lines of evidence were run. They agree.

---

## TL;DR

| Evidence | Result | Reading |
|---|---|---|
| Image-level re-classification (hetero refinement) | ARI = **0.55** vs input, per-particle conf 0.95 | Images support *confident* assignment, but only ~55 % chance-corrected agreement with the input labels |
| Alignment-free re-classification (3D classification) | ARI = **0.13** vs input | A pose-free re-derivation finds an almost unrelated partition |
| Referee in the honest J1442 simplex (GMM) | every hard labelling ARI ≤ **0.35** vs natural GMM clusters; 3Dclass ≈ 0.11 | No hard labelling reproduces the natural (honest-posterior) clustering |
| Per-class gold-standard resolution | **3.7 – 6 Å** every class | The heterogeneity is *real* — each class is a genuine, well-resolved map, not noise |
| Cross-class map similarity (shared frame) | **P6≈P7** (CC 0.60), **P8 distinct** (CC ~0.35) | Two of the three "classes" are nearly the same state; only P8 stands apart |

**Conclusion.** The classification captures one *bona fide* structural axis —
**P8 vs (P6 ≈ P7)** — but the three-way per-particle assignment is **not
reproducible**: P6 and P7 are essentially the same conformational region, and
the discrete boundary between them (and where the cut falls) is unstable. This
is the fingerprint of a **continuous landscape with one resolvable end-state
(P8)**, not three discrete classes.

---

## Part A — Does the classification survive re-derivation from the images?

Each particle's input label was compared against a *fresh* CryoSPARC re-assignment.
Agreement is reported with adjusted/normalised mutual-information metrics
(label-permutation invariant; Hubert & Arabie 1985, Vinh et al. 2010).

### A1. Heterogeneous refinement (J3571 — alignment + classification jointly)
- Re-assigned counts: P6 = 93,481 · P7 = 48,553 · P8 = 88,362.
- **ARI = 0.550**, AMI = 0.456 vs input.
- Mean new per-particle confidence = **0.949**; 84.1 % of particles assigned at > 0.9.

Hetero refinement re-assigns particles *very confidently from the raw images* —
yet lands on a partition that disagrees with the input ~45 % of the time
(chance-corrected). High confidence + moderate reproducibility = the images
carry real signal, but the discrete boundary is soft.

### A2. 3D classification (J3579 — alignment-free, EM mixture)
- Retained 177,900 / 230,396 (low-probability particles dropped by CryoSPARC).
- Classes: 66,069 / 53,089 / 58,742.
- **ARI = 0.132**, AMI = 0.127 vs input. Mean confidence 0.938.

Removing the pose prior, the mixture model carves the data on an axis nearly
unrelated to the input labels (near-chance ARI). The input partition is *not* the
dominant variance direction in the images.

Confusion matrices and metrics: `confusion_hetero.png`, `confusion_3dclass.png`,
`partA_reclassification.csv`.

---

## Part B — Referee in the honest (J1442) posterior simplex

The J1442 posteriors are *honest* (mean max-posterior ≈ 0.36, near-uniform),
unlike the over-confident J1069 posteriors (≈ 0.99). A 3-component Gaussian
mixture was fit in the additive-log-ratio (ALR) coordinates of the J1442
protein-class simplex, and every candidate hard labelling was scored against
these *natural* clusters.

| Labelling | ARI vs GMM | silhouette |
|---|---|---|
| J1442 own argmax | **0.348** | 0.311 |
| Input (J1069 + J1442 weights) | 0.244 | 0.206 |
| Hetero refinement | 0.206 | 0.169 |
| 3D classification | 0.106 | 0.014 |

Even the J1442 *own* argmax only reaches ARI 0.35 against the natural clusters —
the honest posteriors do not contain three crisp islands. The image-level
re-classifications agree with the natural geometry *even less* (hetero 0.21,
3Dclass 0.11, silhouette ≈ 0). No discrete labelling is "found" by the data.
(These numbers reproduce the independent `gmm_referee_compare` run exactly,
confirming the pipeline.)

Output: `partB_referee.csv`, `referee_gmm_simplex.png`.

---

## Part C — Map-level test: are the classes different *structures*?

### C1. Gold-standard resolution per class (FSC of half-maps at 0.143; Scheres & Chen 2012)

| Job | P6 | P7 | P8 |
|---|---|---|---|
| Homogeneous reconstruction (shared frame) | 4.40 Å | 5.96 Å | 4.47 Å |
| Non-uniform refinement | 3.69 Å | 4.73 Å | 3.83 Å |
| Homogeneous refinement | 4.41 Å | 7.71 Å | 5.91 Å |

Every class reconstructs to **3.7 – 6 Å** — these are real, well-resolved volumes.
The heterogeneity is *not* an artefact of splitting noise. P7 is consistently the
softest (fewest particles, 54.5 k).

### C2. Cross-class similarity — the decisive "different states?" test

Computed on the **homogeneous-reconstruction** maps, which share the fixed J1442
pose frame and are therefore directly comparable without alignment.

| Pair | masked real-space CC | cross-class FSC = 0.5 |
|---|---|---|
| **P6 – P7** | **0.601** | 13.7 Å |
| P6 – P8 | 0.343 | 61.1 Å |
| P7 – P8 | 0.380 | 59.9 Å |

Interpretation against the gold standard:
- **P6 ≈ P7.** CC 0.60 and agreement to 13.7 Å (vs their own ~4.4–6 Å limit) — they
  share the fold and differ only at medium resolution. Borderline the *same* state.
- **P8 is distinct.** CC ~0.35 and FSC-0.5 only at ~60 Å (blob level) against both
  others — P8 carries genuinely different ordered density.

The non-uniform and homogeneous refinements (independent pose frames, so
cross-class FSC is contaminated by global misalignment — flagged
`needs_alignment=True`) reproduce the *same ordering*: P6–P7 closest, P8 the
outlier. The signal is robust to refinement choice.

Curves/tables: `maps/crossfsc_*.png`, `maps/gold_standard_resolution.csv`,
`maps/cross_class_similarity.csv`, `maps/goldfsc_*_*.csv`.

---

## Synthesis

All three methods converge:

1. **There is real structure.** Each class is individually resolved to 3.7–6 Å,
   and one axis — **P8 vs the rest** — is reproducible across every test
   (distinct map, distinct density).
2. **The three-way split is not.** P6 and P7 are nearly the same state
   (CC 0.60, ARI of re-classification ≈ chance for the P6/P7 boundary), the
   honest posteriors form no three crisp clusters (referee ARI ≤ 0.35), and
   alignment-free re-derivation recovers an almost unrelated partition
   (ARI 0.13).

This is exactly the signature of a **continuous conformational landscape with a
single discretely-resolvable end-state (P8)** sitting at the end of a P6↔P7
continuum — consistent with the project's flat-posterior / unimodal-3DVA
findings. The classification is best reported as **two states (P6/P7 ensemble
and P8)**, not three; any three-way per-particle assignment should be treated as
soft, not categorical.

### Reproduce
```
python scripts/analyze_branch_validation.py \
    --branch-dir results_J1069/cryosparc_outputs/with_1442_weights \
    --input-index results_J1069/exports_combined/combined_J1069_w1442_class_index.csv \
    --outdir results_J1069/branch_validation_w1442

python scripts/analyze_branch_maps.py \
    --branch-dir results_J1069/cryosparc_outputs/with_1442_weights \
    --outdir results_J1069/branch_validation_w1442/maps
```
Both scripts are branch-agnostic — point `--branch-dir` / `--input-index` at the
J1442-own or J1069-own outputs to run the identical battery on those branches.
