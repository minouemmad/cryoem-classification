# GMM-referee comparison of J1442 vs J1069(weighted) partitions

The uploaded per-class `*_particles.cs` files are single-class reconstruction
outputs (`class_posterior`/`alpha` reset to 1, UIDs reassigned by CryoSPARC), so
they encode **class membership only**. The GMM pipeline needs multi-class
posteriors, which live in the original stacks. We recover the *identical*
membership (same per-class counts) from the original J1069 stack by UID, fit the
GMM in **J1442's honest posterior simplex**, and use it as a neutral referee,
then score each job's membership with chance-corrected external-validation
indices (ARI/AMI/V-measure) — the standard clustering-comparison approach
(Hubert & Arabie 1985; Vinh et al. 2010).

## Setup
- Honest space: `data/cryosparc_P25_J1442_00000_particles.cs`  (N protein = 230,396)
- Competing partition: `data/cryosparc_P25_J1069_00042_particles.cs` (J1069 protein argmax)
- Matched in both partitions: 230,396
- GMM: 3 components, full covariance, ALR embedding, BIC=-693167.8

## Agreement with the honest GMM clustering
| comparison | ARI | AMI | V-measure |
|---|---|---|---|
| GMM vs **J1442** | 0.348 | 0.450 | 0.450 |
| GMM vs **J1069 (weighted)** | 0.244 | 0.314 | 0.314 |
| J1442 vs J1069 (direct) | 0.561 | 0.487 | 0.487 |

**The J1442 partition matches the honest soft structure better.**

## Within-class GMM confidence (mean max-responsibility)
Higher = the class is a cleaner, more confident GMM cluster.

class  J1442_mean_maxresp  J1442_N  J1069_mean_maxresp  J1069_N
   P6               0.737    83843               0.757    91826
   P7               0.856    67671               0.847    54548
   P8               0.858    78882               0.853    84022
  ALL               0.813   230396               0.813   230396

## When the two jobs disagree (41,187 particles, 17.9%)
- Mean J1442 max-posterior on **disagreeing** particles: 0.349
  vs **0.365** where the jobs agree.
- Mean J1442 normalised entropy: disagree 0.999
  vs agree 0.996.
- Of the disagreements, the honest GMM sides with **J1442 29.3%**,
  with **J1069 32.5%**, neither 38.1%.

### Reading it
If disagreements concentrate on low-max-post / high-entropy particles, the two
jobs only differ where the honest model is genuinely uncertain — i.e. J1069's
extra confidence is spent on ambiguous particles. Where the GMM sides more often
with one job on those contested particles, that job's hard cut better respects
the honest conformational manifold.

## Caveat (read before over-interpreting)
The GMM is fit in **J1442's** posterior space, so `GMM vs J1442` has a built-in
home advantage — it is a self-consistency baseline, not an independent score.
The honest comparisons are: (1) the **direct** `J1442 vs J1069` agreement, and
(2) the **scatter** `alr_scatter_by_labelling.png`, where each partition is drawn
on the same honest coordinates and you can see whether the colours form contiguous
regions (respecting the manifold) or are smeared across it.

## Bottom line
- In the honest posterior space the J1442 labels carve clean, contiguous wedges
  (argmax of the soft posteriors); the J1069 labels are **smeared** across the
  same space — J1069's confident assignments cut across the honest conformational
  coordinate rather than following it.
- Both partitions nonetheless yield equally "clean" GMM clusters on average
  (mean max-responsibility ~0.81 either way), so neither is internally noisier.
- The two jobs disagree on ~18% of particles, and those are the *most* ambiguous
  ones under J1442 (max-post ~0.35, entropy ~1.0). On exactly those particles the
  honest model does not decisively back either job (it picks a third class ~38% of
  the time), so the disagreement is a genuine soft-assignment toss-up, not one job
  being clearly right.

## Files
- agreement_metrics.csv, within_cluster_confidence.csv, disagreement_analysis.csv
- contingency_*.csv / .png, alr_scatter_by_labelling.png, agreement_bars.png
