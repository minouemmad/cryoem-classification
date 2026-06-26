# GMM Classification-Uncertainty Pipeline

Implements the GCER classification-uncertainty plan from
`hCFTRCryoEMPipeline05302025a.pdf`,
`Flatiron-cCFTR-HierarchicalUnfoldingPoster07242025b.pdf`, and
`HSHuntScoreClassificationProblem10252024a.pdf`. Built on scikit-learn's
`GaussianMixture`.

## What it needs

A CryoSPARC particle file (`cryosparc_*_particles.cs`) from a
`heterogeneous refinement` job. The only field actually used is

* `alignments3D_multi/class_posterior` — `(N_particles, K_classes)` matrix
  of per-particle class posteriors.

You also tell the pipeline which columns are **dummy** vs. **protein**
classes (either via `--n-dummies` for "first N columns are dummies" or with
`--protein-idx 6 7 8` for explicit indices). For the J1442 file in this
workspace: 9 classes, 6 dummies + 3 protein.

No image data, particle stacks, or volumes are needed — the entire analysis
runs in *probability-assignment space*, which is what the Hunt notes call
for.

## Pipeline steps (mapped to milestones)

| # | Step | Module | Milestone |
|---|------|--------|-----------|
| 0 | Load `.cs`, extract `(N,K)` posterior, derive hard labels from `argmax`. | [data_io.py](gmm_pipeline/data_io.py) | – |
| 1 | Drop dummy classes; keep particles whose hard label is a protein class; renormalize over the `K_protein` protein columns. | [data_io.py](gmm_pipeline/data_io.py) | – |
| 2 | Embed the simplex-valued posteriors in `R^(K_protein-1)` via additive-log-ratio (default) or by dropping one coordinate. | [preprocess.py](gmm_pipeline/preprocess.py) | – |
| 3 | Fit a `GaussianMixture` with `n_components = K_protein`, warm-started from the CryoSPARC hard assignments. Report BIC / AIC / convergence. | [gmm_fit.py](gmm_pipeline/gmm_fit.py) | **M1** |
| 4 | Monte-Carlo confusion matrix `C[i,j] = P(assigned=j | true=i)` by sampling each Gaussian component and re-classifying through the full GMM. Also pairwise Bhattacharyya distance for an analytic overlap bound. | [confusion.py](gmm_pipeline/confusion.py) | **M2** |
| 5 | Bias-corrected populations `pi_true = (Cᵀ)⁻¹ pi_obs`, projected onto the simplex; bootstrap CIs over both particles and refit GMMs. | [uncertainty.py](gmm_pipeline/uncertainty.py) | **M3** |
| 6 | Class-repetition analysis: refit with `K + r` components for `r = 0, 1, 2, …`, map every component to its nearest base class, watch occupancies migrate. | [repetition.py](gmm_pipeline/repetition.py) | **M4 / Hunt strategy 2** |
| 7 | Persist CSV/JSON/NPY + plots into `--outdir`. | [plots.py](gmm_pipeline/plots.py) | – |

## Install & run

```powershell
pip install numpy pandas scipy scikit-learn matplotlib seaborn

python run_pipeline.py `
  --cs data/cryosparc_P25_J1442_00000_particles.cs `
  --n-dummies 6 `
  --transform alr `
  --covariance full `
  --mc-samples 50000 `
  --n-boot 30 `
  --reps 0 1 2 3 `
  --outdir results_J1442
```

Source `.cs` files live in `data/`; standalone utilities (diagnostics,
sanity plots, per-CryoSPARC-class export, GMM bootstrap, re-plotting) live in
`scripts/` and are run from the repo root, e.g. `python scripts/diagnostics/posterior_diagnostics.py`.

Key CLI flags:

* `--protein-idx 6 7 8` — explicit 0-based protein column indices (overrides `--n-dummies`).
* `--transform {alr, drop}` — `alr` is recommended; posteriors hug the simplex boundary so a raw Gaussian fit can be poor.
* `--covariance {full, diag, tied, spherical}` — start with `full`; switch to `diag` if components don't converge or condition numbers blow up.
* `--mc-samples` — samples per component for the confusion matrix.
* `--n-boot` — bootstrap replicates for population CIs.
* `--reps 0 1 2 3` — extra components for class-repetition analysis.

## Outputs (in `--outdir`)

Results are sorted into sub-folders:

```
<outdir>/
  confusion/    all confusion matrices (.csv + .png) + Bhattacharyya overlap
  populations/  observed vs corrected populations, per-matrix comparison, class table
  gmm/          GMM model artifacts, bootstrap arrays, diagnostics, NLL landscape, repetition
  exports/      low-uncertainty particle subsets (.cs/.csv) for downstream refinement
  sanity/       Gaussian-fit sanity plots (written by scripts/diagnostics/make_gaussian_sanity_plots.py)
  refinement/   downstream CryoSPARC refinement results (FSC/maps), added manually
  run.log
```

**`confusion/`** — every confusion matrix (`.csv` + `.png`), ordered by how it
should be used:

* `confusion_soft_posterior` — **primary** matrix for population deconvolution.
  Uses the CryoSPARC posteriors as soft truth; no GMM, no hard-label selection
  bias, no component label-switching. Near rank-1 on flat posteriors (honest
  signal that classes are barely separable).
* `confusion_multiclass_analytical` — score-space Gaussian approximation
  (proper K>2 extension of the erf formula); secondary / model-based estimate.
* `confusion_pairwise_analytical` — pairwise erf rates composed into a full
  matrix via the independence approximation.
* `confusion_montecarlo`, `confusion_gmm_equalprior[_mean/_std]` — GMM
  geometry/separability diagnostics only (component space; not for deconvolution).
* `confusion_empirical` — CryoSPARC-vs-GMM hard-label agreement.
* `class_overlap_bhattacharyya` — `exp(-D_B)` analytic overlap.

**`populations/`**

* `conformational_populations.csv` / `.png` — observed and corrected populations
  with bootstrap CIs (`corrected_soft_posterior` is the primary correction).
* `population_corrections_all_matrices.csv` / `.png` — the corrected population
  **and** diagonal accuracy from *every* confusion matrix, side by side, so you
  can see what each inversion step does to the fractions.
* `summary_class_table.png` — one table per class: observed, the primary 95% CI,
  and each matrix's corrected fraction + accuracy + max overlap.

**`gmm/`**

* `gmm_class_repetition.csv` / `.png` — mapped occupancies vs. duplicate components.
* `gmm_diagnostics.json` — convergence, iters, BIC/AIC, weights, covariance condition numbers.
* `gmm_means_*`, `gmm_weights_*`, `bootstrap_gmm_*.npy`, `gmm_nll_landscape.png`.
* `posterior_protein.npy`, `responsibilities.npy` — raw arrays for downstream work.

## Suggested follow-up workflow

1. Sanity-check `gmm/gmm_diagnostics.json` — `converged: true`, low component condition numbers, BIC stable across runs.
2. Inspect `confusion/confusion_soft_posterior.png` (and the analytical/MC variants). A high diagonal means the CryoSPARC posteriors for that class rarely argmax elsewhere; a near-uniform matrix means the classes are barely separable and the deconvolution will be near-identity.
3. Compare `populations/population_corrections_all_matrices.png` and `summary_class_table.png` to see how each confusion matrix shifts the fractions; the bootstrap error bars on the soft-posterior correction are your reported uncertainty.
4. Use `gmm/responsibilities.npy` to pull a "low-misassignment particle set" (e.g. `max(resp) > 0.9`, already exported to `exports/`) and re-run downstream refinement on that subset.
5. Look at `gmm/gmm_class_repetition.png` — a class whose occupancy stays flat as `r` grows is genuinely populated; a class that bleeds into duplicates likely contains confused particles.
6. Repeat on the larger `gP25W6J1497_*.cs` file to test scaling to more conformational states (Milestone 4).

## Smoke-tested result on `cryosparc_P25_J1442_00000_particles.cs`

```
N=230,396  K=9  protein=[6, 7, 8]  K_protein=3
observed populations (hard, GMM):     [0.284, 0.680, 0.036]
corrected populations (bootstrap):    [0.275 ± 0.011, 0.625 ± 0.007, 0.100 ± 0.014]
diag(C) (Monte-Carlo):                [0.78, 0.90, 0.30]
max pairwise Bhattacharyya overlap:   0.79
```

The third protein class is heavily confused (low diagonal, high overlap), so
its raw 3.6 % occupancy nearly triples after deconvolution — exactly the
kind of bias the Hunt framework is meant to expose.
