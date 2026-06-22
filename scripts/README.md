# Scripts

Standalone tools, grouped by purpose. Run from the repo root, e.g.
`python scripts/maps/compare_maps.py ...`. Scripts that import `gmm_pipeline`
add the repo root to `sys.path` automatically.

`make_proposal.py` (repo-root one-off) generates the Amgen proposal document.

## pipeline/ — GMM pipeline & particle exports

| Script | Purpose |
|--------|---------|
| `run_gmm_bootstrap.py` | Re-fit GMM from saved posteriors; equal-prior + bootstrap confusion CIs |
| `replot_from_csv.py` | Regenerate summary plots from saved CSVs (no refit) |
| `export_threshold_sweep.py` | Export particle subsets across GMM-responsibility thresholds |
| `export_by_cryosparc_class.py` | Export low-uncertainty particles grouped by CryoSPARC class |
| `export_weighted_by_class.py` | Per-class export with soft weight injected into `alignments3D/alpha` |
| `combine_class_stacks.py` | Merge per-class particle stacks |
| `ensemble_gmm_export.py` | Ensemble-GMM export across repeated fits |
| `ensemble_reweight.py` | Ensemble reweighting of populations |

## maps/ — Map comparison & validation

| Script | Purpose |
|--------|---------|
| `align_maps_chimerax.py` | Global rigid alignment (ChimeraX `fitmap` + `resample`) |
| `compare_maps.py` | Real-space CC, FSC curves, difference maps (needs `mrcfile`) |
| `map_density_diagnostics.py` | Core-normalized occupancy + difference localization |
| `analyze_3dva.py` | 3DVA latent-space GMM/BIC vs. hetero-posterior partition |
| `analyze_branch_maps.py` | Compare branch reconstruction maps |
| `analyze_branch_validation.py` | Branch validation against a reference job |

## cryodrgn/ — Continuous heterogeneity

| Script | Purpose |
|--------|---------|
| `cryodrgn_run.py` | Full cryoDRGN workflow: parse -> downsample -> train_vae -> analyze |
| `cryodrgn_compare.py` | Cluster latent z; score vs. CryoSPARC / GMM labels (ARI/NMI/...) |
| `make_synthetic_cryodrgn_demo.py` | Synthetic 2-conformation demo (CPU toolchain test) |

## diagnostics/ — Posterior & model diagnostics

| Script | Purpose |
|--------|---------|
| `posterior_diagnostics.py` | Max-posterior histograms, entropy, per-class violins |
| `make_gaussian_sanity_plots.py` | 1D/2D GMM Gaussian-fit sanity plots |
| `local_followups.py` | Class/dummy occupancy, soft effective-N, continuity test |
| `compare_gmm_spaces.py` | GMM quality: 3DVA-latent vs. hetero-posterior space |
| `compare_uncertainty_models.py` | Five-method uncertainty-model comparison |
| `gmm_referee_compare.py` | Referee/cross-job GMM comparison |
| `plot_crossjob_gaussian_2d.py` | Cross-job 2D Gaussian posterior plot |
| `check_hetero_label_mapping.py` | Verify class/component label mapping |
