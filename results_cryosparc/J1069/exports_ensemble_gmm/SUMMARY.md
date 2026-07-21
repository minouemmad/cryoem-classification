# Ensemble-GMM assignment + CryoSPARC export

This applies the GMM pipeline to the **combined** (ensemble) classifier and
turns the result into particle stacks you can reconstruct.

## What was done
1. Built the calibrated equal-weight ensemble posterior from the two experts:
   J1069 temperature-recalibrated (T*=108.71) to J1442's entropy, then linear pool
   at lambda=0.5 (230,396 shared protein particles).
2. Fit the GMM (k=3, full cov, ALR embedding) to the ensemble
   posteriors. Converged=True, BIC=-805302.3, mean max-responsibility
   0.926.
3. Hard-assigned each particle by GMM argmax and exported disjoint per-class
   stacks, injecting the GMM responsibility as the per-particle scale
   (alignments3D/alpha, combine=replace, beta=1.0).

## Populations
class  ensemble_hard  gmm_hard  ensemble_soft_mean  deconvolved
   P6         0.3910    0.3962              0.3329       0.3329
   P7         0.2489    0.2345              0.3339       0.3339
   P8         0.3601    0.3693              0.3332       0.3332

Ensemble-argmax vs GMM-argmax agreement: 0.981.

## Exported stacks (import + "Homogeneous Reconstruction Only")
| class | N particles | effective N | files (in results_J1069/exports_ensemble_gmm) |
|---|---|---|---|
| P6 | 91,273 | 85791 | `ensembleGMM_P6.cs` + `ensembleGMM_P6_passthrough.cs` |
| P7 | 54,032 | 47064 | `ensembleGMM_P7.cs` + `ensembleGMM_P7_passthrough.cs` |
| P8 | 85,091 | 80398 | `ensembleGMM_P8.cs` + `ensembleGMM_P8_passthrough.cs` |

### How to use in CryoSPARC
1. Import Particle Stack: give each `ensembleGMM_P?.cs` **and** its
   `_passthrough.cs` together.
2. Run **Homogeneous Reconstruction Only** with per-particle scale refinement
   **off** (the scale field already holds the ensemble-GMM weight). Use
   `--combine none` at export time instead if you want CryoSPARC's own weighting.
3. Compare the three maps as in `scripts/compare_maps.py`.

## Files
- ensemble_gmm_populations.csv, ensemble_gmm_assignment_uid.csv,
  ensemble_gmm_responsibilities.npy, export_manifest.csv
- ensembleGMM_P6/P7/P8.cs (+ _passthrough.cs)
