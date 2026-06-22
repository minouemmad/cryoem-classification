# CFTR Cryo-EM Classification-Uncertainty Pipeline

Quantifies **how trustworthy** CryoSPARC heterogeneous-refinement class assignments
are, and corrects conformational populations for class confusion — entirely in
probability-assignment space (no raw particle images required).

The core question: when CryoSPARC assigns a particle to class *i* with some
posterior, how often *would* it land in class *j*? That confusion is used to
(a) report honest per-class populations with uncertainty and (b) export
low-misassignment particle subsets for downstream refinement.

## Repository layout

```
run_pipeline.py        Main entry point: .cs -> GMM -> confusion -> populations -> exports
gmm_pipeline/          Core library (see gmm_pipeline/README.md for the method)
scripts/               Standalone tools, grouped by purpose:
  pipeline/              GMM bootstrap, re-plotting, particle exports, ensembles
  maps/                  Map comparison (CC/FSC), alignment, density diagnostics, 3DVA
  cryodrgn/              cryoDRGN driver, latent-space comparison, synthetic demo
  diagnostics/           Posterior/Gaussian sanity plots, uncertainty-model comparisons
data/                  CryoSPARC .cs inputs, 3DVA latents, and reconstructed maps/
results_J1069/         Per-dataset outputs (confusion/, populations/, gmm/, exports/, ...)
results_J1442/
results_J1497/
results_cryodrgn/      cryoDRGN runs and demos
docs/                  Reference PDFs and generated proposal
diagnostics/           Posterior-quality figures
```

See [docs/WORKFLOW.md](docs/WORKFLOW.md) for the end-to-end workflow and
[scripts/README.md](scripts/README.md) for what each script does.

## Quick start

```powershell
pip install numpy pandas scipy scikit-learn matplotlib seaborn

python run_pipeline.py `
  --cs data/cryosparc_P25_J1442_00000_particles.cs `
  --n-dummies 6 --transform alr --covariance full `
  --mc-samples 50000 --n-boot 30 --reps 0 1 2 3 `
  --outdir results_J1442
```

Only the `alignments3D_multi/class_posterior` field (an `N x K` matrix of
per-particle class posteriors) is used. You specify which columns are dummy vs.
protein classes via `--n-dummies` or `--protein-idx`.

Outputs land in `<outdir>/{confusion,populations,gmm,exports,sanity}/`. The
**primary** result is `confusion/confusion_soft_posterior.*` and the
bootstrap-corrected fractions in `populations/conformational_populations.*`.

## Datasets in this workspace

| Dataset | Classes | Notes |
|---------|---------|-------|
| J1069   | 3 (6 dummy + P6/P7/P8) | Converged hetero-refinement |
| J1442   | 3 (6 dummy + P6/P7/P8) | Converged hetero-refinement; also has 3DVA latents |
| J1497   | 5 (6 dummy + P6–P10)   | 5-class outgroup |

## Documentation

* [gmm_pipeline/README.md](gmm_pipeline/README.md) — method, modules, outputs.
* [docs/WORKFLOW.md](docs/WORKFLOW.md) — full analysis workflow.
* [scripts/README.md](scripts/README.md) — script reference.
