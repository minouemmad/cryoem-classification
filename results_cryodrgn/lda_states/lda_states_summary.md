# LDA-sharpened substate test (J1497, J1442)

**Question.** PCA and F(PC1) maximise *variance*, so a substate differing by a
small local event is invisible — which is why P6/P10 looked identical (map CC
0.91, barrier 0.05 kT). LDA maximises *between-class / within-class* scatter, so
it should expose any systematic feature that separates the "merged" classes.
We decode the LDA endpoints, difference-map them, and test reproducibility by
**re-fitting LDA independently on two disjoint halves** of the particles.

`scripts/cryodrgn/cryodrgn_lda_states.py`, box 64 (apix ~4.15 Å). Endpoints =
5th/95th percentile along the best discriminant axis (conditional-mean z, on
manifold). "medoid CC" = the *typical-particle* class comparison from
decode_states, for reference.

## Results

| pair | Fisher sep (SD) | endpoint CC | medoid CC | FSC=0.5 | split-half diff-map CC |
|------|----|----|----|----|----|
| **J1497 P6–P10** | 0.54 | 0.516 | 0.906 | 14.8 Å | **1.000** |
| **J1497 P8–P9**  | 0.44 | 0.098 | 0.554 | 53 Å | **1.000** |
| J1497 P6–P7 | 1.70 | 0.254 | 0.669 | 53 Å | 0.999 |
| J1442 P6–P7 (control) | 1.61 | 0.239 | 0.756 | 53 Å | 1.000 |

LDA balanced acc: J1497 0.485 (5-class), J1442 0.799 (3-class).

## What this shows

1. **LDA does recover a reproducible density difference for P6↔P10 and P8↔P9.**
   The endpoint CC collapses far below the medoid CC (0.91→0.52, 0.55→0.10), the
   difference map is **localized** central density (not box noise), and it
   reproduces *perfectly* when LDA is trained from scratch on two disjoint
   particle halves (diff-map CC ≈ 1.00). So the distinction is **systematic and
   real in the latent**, not a fitting artifact.
2. **But the classes stay heavily overlapping.** P6/P10 and P8/P9 have Fisher
   separations of only ~0.5 SD (vs ~1.6–1.7 for the genuinely distinct P6/P7).
   The reproducible difference lives in the **tails/extremes** of one continuous,
   overlapping distribution: the 5th-vs-95th-percentile endpoints differ, but a
   *typical* P6 and *typical* P10 particle do not (medoid CC still 0.91).
3. **The difference is low resolution** (FSC=0.5 at 15–53 Å) — a global /
   domain-scale density redistribution, consistent with a continuous
   conformational coordinate, not a fine localized loop/substate.

## Verdict (this is the defensible story)

> P10 is a **structurally distinct substate at the extreme of P6's energetic
> basin**, not an independent state. The hetero-refinement split is backed by a
> reproducible but low-resolution, tail-level density shift along a continuous
> coordinate — *structural heterogeneity without thermodynamic separation*.

`5 reconstructable maps ≠ 5 energetic basins.`

## Important caveat / why focused classification is the right next step

LDA is **supervised on the CryoSPARC labels**, so it is *guaranteed* to find a
direction that separates them — the split-half test proves the difference is
reproducible, but **cannot prove it is independent of whatever signal CryoSPARC
used** (it may amplify the same alignment-driven correlate). The orthogonal
validation is the user's own idea: take only P6+P10 particles, apply a focused
mask around the differing region (the localized central density above), and run
CryoSPARC hetero-refinement K=2. If it reproducibly re-separates them, the
substate is real and reconstructable; if not, the original split was unstable.

Artifacts: `J1497/`, `J1442/` hold `*_lda###.mrc`, `*_lda_substates.png`,
`*_lda_fsc.png`, `*_lda_labels.txt`; `lda_states_metrics.json`.
