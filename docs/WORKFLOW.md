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

## 6. Hybrid pipeline: energetic states vs structural substates

The central result of the cryoDRGN analysis is that the CryoSPARC hetero-refine
classes are **not** all independent states: on the *identical* 230,396 particles,
J1442's 3 classes collapse to 3 free-energy basins but J1497's 5 classes collapse
to only 2-3 basins (P10 shares P6's basin; P9 shares P8's). A single map is not a
single state -- several maps can describe one energetic basin.

The hybrid pipeline reconstructs **both** levels explicitly:

```
230k particles
  -> cryoDRGN latent (z.100.pkl)
  -> F(PC1,PC2) free-energy basins        (energetic states)
  -> assign every particle to a basin
  -> export basin_N_particles.cs           <- scripts/cryodrgn/cryodrgn_basin_occupancy.py --export-basins
  -> CryoSPARC NU-refine each basin        = basin (energetic-state) maps
  -> within-basin focused heteroref K      = structural substates
  -> CryoSPARC NU-refine each substate     = substate maps
```

**Choosing the basin pairings and K (no hand-tuning).** The occupancy matrix
`P(basin | class)` (`results_cryodrgn/basin_occupancy/occupancy_*_hard.csv`)
decides it: assign each class to the basin holding most of its particles; the
classes that share a basin are the substate candidates and `K` = how many share
it. For J1497 this gives Basin 1 = {P6, P10} -> K=2 and Basin 2 = {P7, P8, P9}
-> K=3; for J1442 the matrix is block-diagonal (one class per basin) so K=1 and
the basins recover the classes directly. Do **not** run a global K=5 just to get
five classes -- that imposes the answer; let each basin show whether a split
survives.

**Maps to compare** (rigid-body align first -- see step 3 /
`scripts/cryodrgn/cryodrgn_focused_analysis.py`):
1. basin map vs basin map (are the energetic states distinct?);
2. within-basin substate maps vs each other (do the CryoSPARC sub-classes
   survive from images alone? FSC, local resolution, CC, difference, occupancy);
3. hybrid substate maps vs the original heteroref K=5 maps via a CC matrix
   (does the hybrid reproduce all five? => `5 maps = 2-3 basins x substates`).

The auto-generated plan, exported particle subsets, and occupancy matrices live
in `results_cryodrgn/basin_occupancy/` (`basin_occupancy_summary.md`,
`basin_particles/`). See `results_cryodrgn/J1497_real/RESULTS_SUMMARY.md`
section 7 for the J1497-specific table.

## Key finding

J1442 posteriors are near-uniform (mean max-posterior ≈ 0.36): the classes
genuinely overlap at the particle level. Soft-weighted reconstructions collapse
to one consensus map, so per-conformation separation must come from upstream
image-level classification (hetero / 3DVA / focused), not from reweighting flat
posteriors. The per-class volumes still carry real, localized differences
(P8 has the most extra ordered density).
