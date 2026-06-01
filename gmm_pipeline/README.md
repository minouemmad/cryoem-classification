# GMM Classification-Uncertainty Pipeline

Implements the GCER classification-uncertainty plan from
`hCFTRCryoEMPipeline05302025a.pdf`,
`Flatiron-cCFTR-HierarchicalUnfoldingPoster07242025b.pdf`, and
`HSHuntScoreClassificationProblem10252024a.pdf`. Built on scikit-learn's
`GaussianMixture` (the implementation John recommended).

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
  --cs cryosparc_P25_J1442_00000_particles.cs `
  --n-dummies 6 `
  --transform alr `
  --covariance full `
  --mc-samples 50000 `
  --n-boot 30 `
  --reps 0 1 2 3 `
  --outdir results_J1442
```

Key CLI flags:

* `--protein-idx 6 7 8` — explicit 0-based protein column indices (overrides `--n-dummies`).
* `--transform {alr, drop}` — `alr` is recommended; posteriors hug the simplex boundary so a raw Gaussian fit can be poor.
* `--covariance {full, diag, tied, spherical}` — start with `full`; switch to `diag` if components don't converge or condition numbers blow up.
* `--mc-samples` — samples per component for the confusion matrix.
* `--n-boot` — bootstrap replicates for population CIs.
* `--reps 0 1 2 3` — extra components for class-repetition analysis.

## Outputs (in `--outdir`)

* `confusion_mc.csv` / `.png` — Monte-Carlo misclassification matrix.
* `confusion_empirical.csv` / `.png` — empirical CryoSPARC-vs-GMM hard-label agreement.
* `bhattacharyya_overlap.csv` — `exp(-D_B)` analytic overlap.
* `populations.csv` / `.png` — observed and corrected populations with bootstrap CIs.
* `class_repetition.csv` / `.png` — mapped occupancies vs. number of duplicate components.
* `gmm_diagnostics.json` — convergence, iters, BIC/AIC, weights, per-component covariance condition numbers.
* `posterior_protein.npy`, `responsibilities.npy` — raw arrays for downstream work (e.g. selecting low-misassignment particles).

## Suggested follow-up workflow

1. Sanity-check `gmm_diagnostics.json` — `converged: true`, low component condition numbers, BIC stable across runs.
2. Inspect `confusion_mc.png`. If `diag(C[i,i])` is high (~>0.9), class `i` is well separated.
3. Compare observed vs. corrected populations in `populations.png`; the bootstrap error bars are your reported uncertainty.
4. Use `responsibilities.npy` to pull a "low-misassignment particle set" (e.g. `max(resp) > 0.9`) and re-run downstream refinement / occupancy calculations on that subset.
5. Look at `class_repetition.png` — a class whose occupancy stays flat as `r` grows is genuinely populated; a class that bleeds into duplicates likely contains confused particles.
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
