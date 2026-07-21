# Ensemble reweighting of the J1497 + J1112 classifiers

Same 203,586 CFTR protein particles, classified twice into P6, P7, P8, P9, P10.
Instead of choosing one classifier we combine them into an ensemble posterior and
sweep the mixing weight `lambda` (0 = trust J1112 overconfident, 1 = trust J1497
honest). Two standard rules: **linear opinion pool** (Bayesian model averaging)
and **logarithmic opinion pool** (product of experts). The overconfident expert
(J1112) is also **temperature-recalibrated** (T*=106.99) to match J1497's entropy
before an equal-weight ensemble.

## Inputs / calibration
- J1497 (honest): mean max-post 0.221, entropy 0.998
- J1112 (overconfident): mean max-post 0.992, entropy 0.015
- J1112 recalibrated (T*=106.99): mean max-post 0.229, entropy 0.998

## Populations (soft-mean) at the endpoints and the equal-weight ensemble
| state | J1112 only (lambda=0) | ensemble (lambda=0.5) | J1497 only (lambda=1) |
|---|---|---|---|
| P6 | 0.337 | 0.268 | 0.200 |
| P7 | 0.187 | 0.195 | 0.203 |
| P8 | 0.278 | 0.239 | 0.201 |
| P9 | 0.116 | 0.158 | 0.199 |
| P10 | 0.082 | 0.140 | 0.197 |

Full grid (both pools, hard / soft / deconvolved populations, sharpness, entropy,
GMM agreement) is in `ensemble_populations_vs_lambda.csv`.

## What the sweep shows
- **Sharpness is what lambda controls.** Mean max-posterior glides smoothly from
  ~0.99 (J1112) to ~0.22 (J1497)
  (`sharpness_vs_lambda.png`); the global conformational mixture moves far less.
- **An overconfident expert can hijack the ensemble's HARD decisions** in the raw
  linear pool, because near one-hot posteriors dominate the argmax even at small
  weight (`gmm_agreement_vs_lambda.png`).
- **Temperature recalibration** rescales J1112 to J1497's entropy (T*=107.0) before
  an equal-weight ensemble: hard-label agreement with the honest GMM =
  ARI 0.407, entropy 0.998.
  Calibrate experts *before* pooling.
- **The log pool collapses toward the overconfident expert** for small lambda
  (product-of-experts veto), so linear-on-calibrated is the safe default.

## Practical recommendation
- For **population reporting**, report the equal-weight calibrated ensemble:
  P6, P7, P8, P9, P10 = 0.202 / 0.202 / 0.202 / 0.198 / 0.196.
- For **reconstruction weights**, prefer the calibrated equal-weight ensemble
  (`ensemble_posterior_linearcal_l0.5_uid.npz`) over the raw one. Feed these
  per-particle posteriors as `--weights-cs`-style inputs to
  `export_weighted_by_class.py` for a principled middle ground.

## Files
- ensemble_populations_vs_lambda.csv
- populations_vs_lambda_linear.png, populations_vs_lambda_log.png
- sharpness_vs_lambda.png, gmm_agreement_vs_lambda.png
- ensemble_posterior_linear_l0.5_uid.npz, ensemble_posterior_linearcal_l0.5_uid.npz
