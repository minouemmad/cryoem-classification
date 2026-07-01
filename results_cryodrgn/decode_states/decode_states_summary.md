# Decode-states: per-class cryoDRGN volumes + map cross-correlation

For each CryoSPARC class we decode a real cryoDRGN volume at the class latent
**medoid** (the on-manifold particle closest to the class mean z), using the
local `weights.100.pkl` + `config.yaml`, at box 64 for speed. We then compute the
masked real-space cross-correlation (CC) between the per-class maps.

- **High CC (~0.9)** => the two classes decode to essentially the *same density*
  (a substate, not a distinct state).
- **Low CC (<0.4)** => the hetero-refinement split corresponds to a *real*
  structural difference.

This is the **structural** complement to the energetic density-break test in
`results_cryodrgn/class_trajectories/`. The two are independent: one asks "is
there an energy barrier between the classes?", the other "do they decode to
different maps?".

## J1442 (3 classes)

| CC | P6 | P7 | P8 |
|----|----|----|----|
| P6 | 1.00 | 0.76 | 0.14 |
| P7 | 0.76 | 1.00 | 0.12 |
| P8 | 0.14 | 0.12 | 1.00 |

- P6 and P7 are structurally similar (0.76); **P8 is the lone distinct map**.
- Matches the energetics exactly: P6-P7 barrier 0.37 kT (no break), P8 split off
  by 1.3-3.0 kT.

## J1497 (5 classes)

| CC | P6 | P7 | P8 | P9 | P10 |
|----|----|----|----|----|----|
| P6 | 1.00 | 0.67 | 0.19 | 0.56 | **0.91** |
| P7 | 0.67 | 1.00 | 0.12 | **0.84** | **0.90** |
| P8 | 0.19 | 0.12 | 1.00 | 0.57 | 0.12 |
| P9 | 0.56 | 0.84 | 0.57 | 1.00 | 0.72 |
| P10 | **0.91** | **0.90** | 0.12 | 0.72 | 1.00 |

- **P10 ≡ P6 (CC 0.91)** and P10 ≈ P7 (0.90): P10 is a near-duplicate of the
  P6/P7 density. Confirms the energetics (P6-P10 barrier 0.05 kT, P7-P10 0.07).
- **P8 is the structural outlier** (CC ≤ 0.57 to every other class).
- **P9 is a structural intermediate**: closest to P7 (0.84), then P10 (0.72),
  then P8/P6 (~0.56). Energetically P9 shares a basin with P8 (barrier 0.04 kT)
  yet structurally it leans toward the P7 group — i.e. P9 bridges the two ends.

## Convergent conclusion (both analyses, both datasets)

1. The CryoSPARC classes are **not 5 (or 3) independent structures**. The decoded
   maps collapse into **two structural endpoints** — a P6/P7/P10 group and P8 —
   with P9 an intermediate.
2. **P10 is a redundant copy of P6** by every measure (latent overlap 0.61,
   energy barrier 0.05 kT, map CC 0.91).
3. J1442 (3-class) and J1497 (5-class), run on the *identical* 230,396 particles,
   recover the *same* underlying 1-D landscape: a continuous P6 → P7 → P8 axis
   whose only genuinely distinct endpoints are the P6-like and P8 conformations.
4. The "5 states" from hetero-refinement are an **over-partition of a largely
   continuous landscape**: extra classes subdivide populated regions (P6→P6+P10,
   P7→P7+P9) rather than discovering new structures.

Artifacts: `J1442/` and `J1497/` hold the decoded `*_vol###.mrc` (vol001 = first
class, in `*_zfile_labels.txt` order; trajectory volumes follow the class
centroids), `decode_states_cc.png`, `decode_states_metrics.json`.
