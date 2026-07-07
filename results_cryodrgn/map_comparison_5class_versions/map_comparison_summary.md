# Map comparison: cryoDRGN basins vs CryoSPARC classes

Every map reconstructed in its own pose frame was rigid-body aligned to a common reference before any metric. FSC crossings are the *resolution of agreement* between two maps (lower A = more similar). 'Local agreement' is a sliding-window correlation between a matched pair (green=agree, red=differ); true per-voxel local *resolution* needs half-maps, which were not provided.

## V3_J1442basin

- working box 192, apix 1.383 A, physical field 265.6 A
- each cryoDRGN/subhetero map was pairwise rigid-aligned onto *each* CryoSPARC class; the match is the class with the best FSC-based agreement score (not raw CC).

### Cross-method assignment (best CryoSPARC match)

| map | best match | score | CC(lp) | FSC0.143 (A) | rot (deg) | runner-up (score) |
|---|---|---|---|---|---|---|
| cryodrgn:J3815 | **P8** | 0.52 | 0.94 | 8.3 | 163 | P6 (0.36) |
| cryodrgn:J3818 | **P6** | 0.95 | 1.00 | 3.1 | 1 | P7 (0.42) |
| cryodrgn:J3819 | **P10** | 0.86 | 0.99 | 4.3 | 4 | P7 (0.28) |
| cryodrgn:J3820 | **P8** | 0.95 | 1.00 | 3.2 | 1 | P7 (0.38) |
| cryodrgn:J3821 | **P9** | 0.90 | 0.99 | 3.9 | 2 | P8 (0.28) |

### Matched-pair differences

| pair | CC | FSC0.5 (A) | FSC0.143 (A) | diff RMS | local-agree median | local-agree p10 |
|---|---|---|---|---|---|---|
| V3_J1442basin_J3815_vs_P8 | 0.940 | 14.8 | 8.3 | 1.10 | 0.03 | -0.03 |
| V3_J1442basin_J3818_vs_P6 | 0.998 | 3.8 | 3.1 | 0.62 | 0.15 | 0.05 |
| V3_J1442basin_J3819_vs_P10 | 0.989 | 7.4 | 4.3 | 0.69 | 0.14 | 0.01 |
| V3_J1442basin_J3820_vs_P8 | 0.997 | 3.9 | 3.2 | 0.60 | 0.24 | 0.12 |
| V3_J1442basin_J3821_vs_P9 | 0.992 | 6.2 | 3.9 | 0.62 | 0.13 | 0.02 |

## V2_J1497basin

- working box 192, apix 1.383 A, physical field 265.6 A
- each cryoDRGN/subhetero map was pairwise rigid-aligned onto *each* CryoSPARC class; the match is the class with the best FSC-based agreement score (not raw CC).

### Cross-method assignment (best CryoSPARC match)

| map | best match | score | CC(lp) | FSC0.143 (A) | rot (deg) | runner-up (score) |
|---|---|---|---|---|---|---|
| cryodrgn:b1_P6 | **P6** | 0.94 | 1.00 | 3.2 | 1 | P7 (0.45) |
| cryodrgn:b1_P10 | **P10** | 0.87 | 0.99 | 4.2 | 3 | P7 (0.34) |
| cryodrgn:b2_P7 | **P7** | 0.94 | 1.00 | 3.8 | 2 | P6 (0.41) |
| cryodrgn:b2_P8 | **P9** | 0.90 | 0.99 | 4.0 | 2 | P6 (0.26) |
| cryodrgn:b2_P9 | **P8** | 0.96 | 1.00 | 3.2 | 1 | P7 (0.39) |

### Matched-pair differences

| pair | CC | FSC0.5 (A) | FSC0.143 (A) | diff RMS | local-agree median | local-agree p10 |
|---|---|---|---|---|---|---|
| V2_J1497basin_b1_P6_vs_P6 | 0.998 | 3.9 | 3.2 | 0.68 | 0.09 | 0.00 |
| V2_J1497basin_b1_P10_vs_P10 | 0.991 | 7.4 | 4.2 | 0.59 | 0.18 | 0.01 |
| V2_J1497basin_b2_P7_vs_P7 | 0.998 | 5.9 | 3.8 | 0.49 | 0.24 | 0.09 |
| V2_J1497basin_b2_P8_vs_P9 | 0.993 | 6.3 | 4.0 | 0.61 | 0.13 | 0.02 |
| V2_J1497basin_b2_P9_vs_P8 | 0.998 | 3.9 | 3.2 | 0.62 | 0.23 | 0.13 |
