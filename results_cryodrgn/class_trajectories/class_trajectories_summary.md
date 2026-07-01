# cryoDRGN per-class trajectories + density-break test

For each CryoSPARC class pair we project both clouds onto the line joining their latent centroids and measure the free-energy barrier `F=-log p` between them. **barrier < ~0.5 kT** = smooth interpolation / one basin (substates); **barrier > ~1 kT** = a real density break (distinct states). `overlap` is the 1-D distribution overlap coefficient (1 = identical).

## J1442 (3-class, 230,396 particles)

- PC1 15.5% / PC2 11.9% latent variance.
- **Pathway (by mean PC1): P6 -> P7 -> P8**.
- per-class PC1 mean (n): P6 -1.15 (83,843), P7 +0.04 (67,671), P8 +1.19 (78,882).

  Density-break barrier (kT), upper triangle:

  | | P6 | P7 | P8 |
  |---|---|---|---|
  | P6 | - | 0.37 | 2.99 |
  | P7 | 0.37 | - | 1.34 |
  | P8 | 2.99 | 1.34 | - |

  - **No density break (<0.5 kT, same basin):** P6-P7 (0.37).
  - **Real density break (>=1 kT):** P6-P8 (2.99), P7-P8 (1.34).

## J1497 (5-class, 230,396 particles)

- PC1 15.5% / PC2 11.2% latent variance.
- **Pathway (by mean PC1): P6 -> P10 -> P7 -> P9 -> P8**.
- per-class PC1 mean (n): P6 -1.13 (66,134), P7 +0.06 (48,318), P8 +1.22 (53,397), P9 +0.81 (34,075), P10 -0.73 (28,472).

  Density-break barrier (kT), upper triangle:

  | | P6 | P7 | P8 | P9 | P10 |
  |---|---|---|---|---|---|
  | P6 | - | 0.40 | 2.03 | 1.53 | 0.05 |
  | P7 | 0.40 | - | 3.22 | 0.83 | 0.07 |
  | P8 | 2.03 | 3.22 | - | 0.04 | 1.03 |
  | P9 | 1.53 | 0.83 | 0.04 | - | 1.30 |
  | P10 | 0.05 | 0.07 | 1.03 | 1.30 | - |

  - **No density break (<0.5 kT, same basin):** P6-P7 (0.40), P6-P10 (0.05), P7-P10 (0.07), P8-P9 (0.04).
  - **Real density break (>=1 kT):** P6-P8 (2.03), P6-P9 (1.53), P7-P8 (3.22), P8-P10 (1.03), P9-P10 (1.30).
