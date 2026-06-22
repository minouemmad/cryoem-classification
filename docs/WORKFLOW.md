# Analysis Workflow

End-to-end path from a CryoSPARC heterogeneous-refinement job to
confusion-corrected populations, validated reconstructions, and (optionally)
continuous-heterogeneity analysis. Run everything from the repo root.

## 1. Core classification-uncertainty pipeline

```powershell
python run_pipeline.py --cs data/cryosparc_P25_J1442_00000_particles.cs `
  --n-dummies 6 --transform alr --covariance full `
  --mc-samples 50000 --n-boot 30 --reps 0 1 2 3 --outdir results_J1442
```

Produces, in `results_J1442/`:

* `confusion/` — confusion matrices. **Primary:** `confusion_soft_posterior`.
* `populations/` — observed vs. bootstrap-corrected conformational populations.
* `gmm/` — GMM model, diagnostics, responsibilities, repetition analysis.
* `exports/` — low-misassignment particle subsets (`max(resp) > threshold`).

See [gmm_pipeline/README.md](../gmm_pipeline/README.md) for the method and every output file.

## 2. Particle exports for downstream refinement

| Goal | Script |
|------|--------|
| Export by GMM responsibility threshold (sweep) | `scripts/pipeline/export_threshold_sweep.py` |
| Export grouped by CryoSPARC class | `scripts/pipeline/export_by_cryosparc_class.py` |
| Soft/weighted per-class export (inject weight into `alignments3D/alpha`) | `scripts/pipeline/export_weighted_by_class.py` |
| Combine class stacks | `scripts/pipeline/combine_class_stacks.py` |
| Ensemble GMM / reweighting | `scripts/pipeline/ensemble_gmm_export.py`, `ensemble_reweight.py` |
| Re-fit GMM + bootstrap from saved posteriors | `scripts/pipeline/run_gmm_bootstrap.py` |
| Re-plot from saved CSVs | `scripts/pipeline/replot_from_csv.py` |

Reconstruct the exported subsets in CryoSPARC, then compare the maps (step 3).

## 3. Map comparison and validation

Independent refinements are **not** in a common frame — always align before
comparing.

```powershell
# Align (ChimeraX fitmap + resample), then compare CC / FSC / difference maps
python scripts/maps/align_maps_chimerax.py --maps <m1.mrc> <m2.mrc> --bin 3.32 --search 80
python scripts/maps/compare_maps.py --maps <aligned...> --labels P6 P7 P8 --outdir <out>
python scripts/maps/map_density_diagnostics.py --maps <aligned...> --outdir <out>
```

`compare_maps.py` and `map_density_diagnostics.py` need `mrcfile` (use the
system `python`, not the `.venv`).

## 4. Continuous heterogeneity (optional)

* **3DVA:** `scripts/maps/analyze_3dva.py` — latent-space GMM/BIC vs. the hetero
  posteriors.
* **cryoDRGN:** `scripts/cryodrgn/cryodrgn_run.py` (parse -> downsample ->
  train_vae -> analyze) then `scripts/cryodrgn/cryodrgn_compare.py` to score the
  latent clustering against the CryoSPARC / GMM labels. Real training needs a GPU
  and the raw particle images; `make_synthetic_cryodrgn_demo.py` validates the
  toolchain on CPU.

## 5. Diagnostics

`scripts/diagnostics/` holds posterior-quality plots
(`posterior_diagnostics.py`), Gaussian-fit sanity checks
(`make_gaussian_sanity_plots.py`), uncertainty-model and GMM-space comparisons,
and label-mapping checks.

## Key finding

J1442 posteriors are near-uniform (mean max-posterior ≈ 0.36): the classes
genuinely overlap at the particle level. Soft-weighted reconstructions collapse
to one consensus map, so per-conformation separation must come from upstream
image-level classification (hetero / 3DVA / focused), not from reweighting flat
posteriors. The per-class volumes still carry real, localized differences
(P8 has the most extra ordered density).
