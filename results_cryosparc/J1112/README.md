# J1112 Cross-Job Analysis (J1112 assignment + J1497 posteriors)

This directory contains cross-job comparison and export outputs for the J1112/J1497 analysis — an analog of the J1069/J1442 study.

## Context
- **J1112**: 5-class heterogeneous refinement (parent job), 253,835 particles, mean max-posterior **0.99** (overconfident, like J1069)
- **J1497**: Single E-step honest posteriors applied to the same particles, mean max-posterior **0.22** (honest, like J1442)
- **Shared particles**: 203,586 (both protein-argmax; neither a strict subset)

Source .cs files: `data/J1112.particles_all_classes.*.cs` and `data/gP25W6J1497_00000_particles.cs`

## Directory Structure

### `gmm_referee/`
**"What does J1112's hard assignment look like in J1497's honest posterior space?"**

Fits a GMM in J1497's 5-class honest posterior simplex and evaluates both partitions:
- **agreement_metrics.csv** — ARI / AMI / NMI / V-measure for GMM-vs-J1497, GMM-vs-J1112, direct J1497-vs-J1112
- **contingency_*.csv/.png** — row-normalized confusion tables and heatmaps
- **within_cluster_confidence.csv** — mean GMM max-responsibility per class (cluster cleanliness)
- **disagreement_analysis.csv** — where the two jobs disagree; which does the honest GMM side with
- **alr_scatter_by_labelling.png** — ALR 2D projection colored by each partition
- **agreement_bars.png** — headline ARI/AMI/V-measure comparison
- **SUMMARY.md** — interpretation and findings

**Key finding**: GMM-vs-J1497 ARI ≈ 0.30 ≈ GMM-vs-J1112 ARI (nearly tied, unlike J1442/J1069 0.348 > 0.244). On disagreements, the honest GMM sides with **J1112 41.5%** over J1497 29.4%, because J1497's single E-step posteriors are near-uniform (entropy 0.998), making hard argmax noisier than J1112's converged hetero labels. **Opposite** of the J1069/J1442 finding.

### `ensemble_reweight/`
**Sweep mixing weight λ ∈ [0,1] between the two classifiers.**

Linear opinion pool: r_ens = λ * r_J1497 + (1 - λ) * r_J1112 (and logarithmic variant).
Includes temperature recalibration of J1112 to match J1497 entropy (T*≈107, mirroring J1069/J1442's 108.7).

- **ensemble_populations_vs_lambda.csv** — full sweep table (hard, soft, deconvolved populations; sharpness; entropy; GMM agreement)
- **populations_vs_lambda_{linear,log}.png** — population trajectories across the sweep
- **sharpness_vs_lambda.png** — mean max-posterior and entropy vs λ
- **gmm_agreement_vs_lambda.png** — ARI vs honest GMM clustering vs λ
- **ensemble_posterior_linear_l0.5_uid.npz** — raw equal-weight ensemble posteriors (by uid)
- **ensemble_posterior_linearcal_l0.5_uid.npz** — temperature-recalibrated equal-weight ensemble
- **SUMMARY.md** — lessons and recommendation for use

### `exports_weighted/`
**Per-class particle stacks for CryoSPARC reconstruction.**

Hard subsets (disjoint by J1112 argmax) weighted by J1497 honest posteriors.

- **hardJ1112_w1497_b1_P{6..10}.cs** — particle stack (image refs from J1112)
- **hardJ1112_w1497_b1_P{6..10}_passthrough.cs** — alignment + CTF + **blob** (from J1112.particles_all_classes.blob.cs), with `alignments3D/alpha` set to J1497 class posterior weight
- **hardJ1112_w1497_export_manifest.csv** — metadata (counts, effective-N, mean weights)

**Import in CryoSPARC**: Each `_b1_P{k}.cs` + `_passthrough.cs` pair → **Homogeneous Reconstruction Only** (per-particle scale refinement OFF; alpha already holds the posterior weight).

Counts and effective-N (sum of weights):
| Class | n_particles | effective_N |
|-------|-------------|-------------|
| P6    | 68,559      | 15,056      |
| P7    | 37,999      | 8,179       |
| P8    | 56,654      | 12,532      |
| P9    | 23,599      | 5,164       |
| P10   | 16,775      | 3,636       |

Note: effective-N << counts because J1497 posteriors are near-uniform (~0.20 per class, entropy 0.998).

## Scripts & Reproducibility

All scripts are now job-agnostic and can be reused for similar cross-job analyses:

```bash
# Referee
$env:PYTHONPATH="."
python scripts/diagnostics/gmm_referee_compare.py \
  --honest-cs data/gP25W6J1497_00000_particles.cs \
  --assign-cs data/J1112.particles_all_classes.alignments3D_multi.cs \
  --honest-name J1497 --assign-name J1112 \
  --n-dummies 6 --outdir results_J1112/gmm_referee

# Ensemble reweight
python scripts/pipeline/ensemble_reweight.py \
  --honest-cs data/gP25W6J1497_00000_particles.cs \
  --assign-cs data/J1112.particles_all_classes.alignments3D_multi.cs \
  --honest-name J1497 --assign-name J1112 \
  --n-dummies 6 --outdir results_J1112/ensemble_reweight

# Export weighted
python scripts/pipeline/export_weighted_by_class.py \
  --cs data/J1112.particles_all_classes.alignments3D_multi.cs \
  --weights-cs data/gP25W6J1497_00000_particles.cs \
  --assign-cs data/J1112.particles_all_classes.alignments3D_multi.cs \
  --passthrough-cs data/J1112.particles_all_classes.blob.cs \
  --n-dummies 6 --beta 1 --name-prefix hardJ1112_w1497 \
  --outdir results_J1112
```

## Next Steps

1. **Import exports to CryoSPARC**: Each weighted per-class set + passthrough → Homogeneous Reconstruction Only
2. **Compare per-class maps**: Use `scripts/compare_maps.py` to assess class separation (CC/FSC) after reconstruction
3. **Optional**: Run higher β values (e.g., `--beta 2 4 8`) for sharpening sweep and alternative reconstructions
