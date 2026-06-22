# Ensemble reweighting of the J1442 + J1069 classifiers

Same 230,396 CFTR protein particles, classified twice into P6/P7/P8. Instead of
choosing one classifier we combine them into an ensemble posterior and sweep the
mixing weight `lambda` (0 = trust J1069 overconfident, 1 = trust J1442 honest).
Two standard rules: **linear opinion pool** (Bayesian model averaging) and
**logarithmic opinion pool** (product of experts). The overconfident expert
(J1069) is also **temperature-recalibrated** (T*=108.71) to match J1442's entropy
before an equal-weight ensemble.

## Inputs / calibration
- J1442 (honest): mean max-post 0.362, entropy 0.997
- J1069 (overconfident): mean max-post 0.992, entropy 0.021
- J1069 recalibrated (T*=108.71): mean max-post 0.368, entropy 0.997

## Populations (soft-mean) at the endpoints and the equal-weight ensemble
| state | J1069 only (lambda=0) | ensemble (lambda=0.5) | J1442 only (lambda=1) |
|---|---|---|---|
| P6 | 0.398 | 0.364 | 0.331 |
| P7 | 0.237 | 0.287 | 0.337 |
| P8 | 0.365 | 0.348 | 0.332 |

Full grid (both pools, hard / soft / deconvolved populations, sharpness, entropy,
GMM agreement) is in `ensemble_populations_vs_lambda.csv`.

## What the sweep shows
- **Populations are robust; sharpness is what lambda controls.** Soft-mean P6/P7/P8
  move only a few percent across the whole sweep, while mean max-posterior glides
  smoothly from ~0.99 (J1069) to ~0.36 (J1442)
  (`sharpness_vs_lambda.png`). The global conformational mixture is insensitive to which
  expert you trust; only the per-particle decisiveness changes.
- **An overconfident expert hijacks the ensemble's HARD decisions.** In the linear pool
  the hard-label agreement with the honest GMM is pinned at J1069's value (ARI 0.244) for
  *every* lambda from 0 to 0.90, and only jumps (to 0.348) at lambda=1 (`gmm_agreement_vs_lambda.png`).
  Because J1069's posteriors are near one-hot, the term `(1-lambda) * r_J1069` dominates the
  argmax unless lambda is essentially 1 - so naive averaging inherits J1069's bias on the
  decisions that matter for reconstruction, even at small weight.
- **Temperature recalibration fixes this.** Rescaling J1069 to J1442's entropy (T*=108.7)
  before an equal-weight ensemble lifts hard-label agreement to ARI=0.273
  and yields balanced, honest populations (entropy 0.997).
  Calibrate the experts *before* pooling, not after.
- **The log pool collapses toward the overconfident expert** for small lambda (any near-zero
  posterior vetoes a class - the product-of-experts failure mode), so the linear pool on
  *calibrated* inputs is the safe default.

## Practical recommendation
- For **population reporting**, the answer is insensitive to lambda - report the equal-weight
  calibrated ensemble: P6/P7/P8 = 0.333 /
  0.334 / 0.333
  (deconvolved values agree to <1%). This matches J1442 to within a couple of percent.
- For **reconstruction weights**, prefer the calibrated equal-weight ensemble
  (`ensemble_posterior_linearcal_l0.5_uid.npz`) over the raw one - the raw linear ensemble
  would still hard-assign like J1069. Feed these per-particle posteriors as `--weights-cs`-style
  inputs to `export_weighted_by_class.py` for a principled middle ground between J1069's hard
  cut and J1442's soft averaging.

## Files
- ensemble_populations_vs_lambda.csv
- populations_vs_lambda_linear.png, populations_vs_lambda_log.png
- sharpness_vs_lambda.png, gmm_agreement_vs_lambda.png
- ensemble_posterior_linear_l0.5_uid.npz, ensemble_posterior_linearcal_l0.5_uid.npz
