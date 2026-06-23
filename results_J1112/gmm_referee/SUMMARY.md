# GMM-referee comparison of J1497 vs J1112 partitions

The same particles are classified two ways. J1112 is the (over)confident
assignment job; J1497 is the honest soft refinement. We fit a GMM in **J1497's
honest posterior simplex** and use it as a neutral referee, then score each
job's hard membership with chance-corrected external-validation indices
(ARI/AMI/V-measure) — the standard clustering-comparison approach
(Hubert & Arabie 1985; Vinh et al. 2010).

## Setup
- Honest referee space: `data/gP25W6J1497_00000_particles.cs`  (N protein = 230,396)
- Competing partition: `data/J1112.particles_all_classes.alignments3D_multi.cs` (J1112 protein argmax)
- Matched in both partitions: 203,586
- GMM: 5 components (P6, P7, P8, P9, P10), full covariance, ALR embedding, BIC=-1853678.8

## Agreement with the honest GMM clustering
| comparison | ARI | AMI | V-measure |
|---|---|---|---|
| GMM vs **J1497** | 0.299 | 0.317 | 0.317 |
| GMM vs **J1112** | 0.296 | 0.261 | 0.261 |
| J1497 vs J1112 (direct) | 0.500 | 0.448 | 0.448 |

**The J1497 partition matches the honest soft structure better.**

## Within-class GMM confidence (mean max-responsibility)
Higher = the class is a cleaner, more confident GMM cluster.

class  J1497_mean_maxresp  J1497_N  J1112_mean_maxresp  J1112_N
   P6               0.851    59995               0.821    68559
   P7               0.609    41258               0.627    37999
   P8               0.808    49656               0.787    56654
   P9               0.740    29085               0.753    23599
  P10               0.689    23592               0.695    16775
  ALL               0.757   203586               0.757   203586

## When the two jobs disagree (52,844 particles, 26.0%)
- Mean J1497 max-posterior on **disagreeing** particles: 0.215
  vs **0.224** where the jobs agree.
- Mean J1497 normalised entropy: disagree 0.998
  vs agree 0.998.
- Of the disagreements, the honest GMM sides with **J1497 29.4%**,
  with **J1112 41.5%**, neither 29.1%.

### Reading it
If disagreements concentrate on low-max-post / high-entropy particles, the two
jobs only differ where the honest model is genuinely uncertain — i.e. J1112's
extra confidence is spent on ambiguous particles. Where the GMM sides more often
with one job on those contested particles, that job's hard cut better respects
the honest conformational manifold.

## Caveat (read before over-interpreting)
The GMM is fit in **J1497's** posterior space, so `GMM vs J1497` has a built-in
home advantage — it is a self-consistency baseline, not an independent score.
The honest comparisons are: (1) the **direct** `J1497 vs J1112` agreement, and
(2) the **scatter** `alr_scatter_by_labelling.png`, where each partition is drawn
on the same honest coordinates and you can see whether the colours form contiguous
regions (respecting the manifold) or are smeared across it.

## Files
- agreement_metrics.csv, within_cluster_confidence.csv, disagreement_analysis.csv
- contingency_*.csv / .png, alr_scatter_by_labelling.png, agreement_bars.png
