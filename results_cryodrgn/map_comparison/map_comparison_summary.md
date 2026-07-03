# Map comparison: cryoDRGN basins vs CryoSPARC classes

Every map reconstructed in its own pose frame was rigid-body aligned to a common reference before any metric. FSC crossings are the *resolution of agreement* between two maps (lower A = more similar). 'Local agreement' is a sliding-window correlation between a matched pair (green=agree, red=differ); true per-voxel local *resolution* needs half-maps, which were not provided.

## J1442

- working box 192, apix 1.383 A, physical field 265.6 A
- each cryoDRGN/subhetero map was pairwise rigid-aligned onto *each* CryoSPARC class; the match is the class with the best FSC-based agreement score (not raw CC).

### Cross-method assignment (best CryoSPARC match)

| map | best match | score | CC(lp) | FSC0.143 (A) | rot (deg) | runner-up (score) |
|---|---|---|---|---|---|---|
| cryodrgn:basin1 | **P7** | 0.44 | 0.94 | 10.6 | 6 | P6 (0.34) |
| cryodrgn:basin2 | **P6** | 0.75 | 0.97 | 5.1 | 12 | P8 (0.35) |
| cryodrgn:basin3 | **P7** | 0.45 | 0.92 | 10.2 | 178 | P8 (0.25) |
| subhetero:sub2982 | **P8** | 0.56 | 0.87 | 6.6 | 20 | P6 (0.46) |
| subhetero:sub2984 | **P7** | 0.66 | 0.94 | 7.0 | 13 | P8 (0.26) |

### Matched-pair differences

| pair | CC | FSC0.5 (A) | FSC0.143 (A) | diff RMS | local-agree median | local-agree p10 |
|---|---|---|---|---|---|---|
| J1442_basin1_vs_P7 | 0.940 | 17.7 | 10.6 | 1.14 | 0.05 | -0.02 |
| J1442_basin2_vs_P6 | 0.975 | 8.0 | 5.1 | 0.90 | 0.05 | -0.02 |
| J1442_basin3_vs_P7 | 0.917 | 17.7 | 10.2 | 1.13 | 0.07 | -0.02 |
| J1442_sub2982_vs_P8 | 0.873 | 19.0 | 6.6 | 0.95 | 0.04 | -0.02 |
| J1442_sub2984_vs_P7 | 0.941 | 8.6 | 7.0 | 0.89 | 0.06 | -0.03 |

## J1497

- working box 192, apix 1.383 A, physical field 265.6 A
- each cryoDRGN/subhetero map was pairwise rigid-aligned onto *each* CryoSPARC class; the match is the class with the best FSC-based agreement score (not raw CC).

### Cross-method assignment (best CryoSPARC match)

| map | best match | score | CC(lp) | FSC0.143 (A) | rot (deg) | runner-up (score) |
|---|---|---|---|---|---|---|
| cryodrgn:b1_P6 | **P6** | 0.94 | 1.00 | 3.2 | 1 | P7 (0.45) |
| cryodrgn:b1_P10 | **P10** | 0.87 | 0.99 | 4.1 | 3 | P7 (0.34) |
| cryodrgn:b2_P7 | **P7** | 0.94 | 1.00 | 3.8 | 2 | P6 (0.41) |
| cryodrgn:b2_P8 | **P9** | 0.90 | 0.99 | 4.0 | 2 | P7 (0.25) |
| cryodrgn:b2_P9 | **P8** | 0.96 | 1.00 | 3.2 | 1 | P7 (0.39) |

### Matched-pair differences

| pair | CC | FSC0.5 (A) | FSC0.143 (A) | diff RMS | local-agree median | local-agree p10 |
|---|---|---|---|---|---|---|
| J1497_b1_P6_vs_P6 | 0.998 | 3.9 | 3.2 | 0.68 | 0.09 | 0.00 |
| J1497_b1_P10_vs_P10 | 0.991 | 7.4 | 4.1 | 0.59 | 0.18 | 0.01 |
| J1497_b2_P7_vs_P7 | 0.998 | 5.9 | 3.8 | 0.49 | 0.24 | 0.09 |
| J1497_b2_P8_vs_P9 | 0.993 | 6.3 | 4.0 | 0.61 | 0.13 | 0.02 |
| J1497_b2_P9_vs_P8 | 0.998 | 3.9 | 3.2 | 0.62 | 0.23 | 0.13 |
