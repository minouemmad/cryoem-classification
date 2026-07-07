# Map comparison: cryoDRGN basins vs CryoSPARC classes

Every map reconstructed in its own pose frame was rigid-body aligned to a common reference before any metric. FSC crossings are the *resolution of agreement* between two maps (lower A = more similar). 'Local agreement' is a sliding-window correlation between a matched pair (green=agree, red=differ); true per-voxel local *resolution* needs half-maps, which were not provided.

## J1442basin_vs_best

- working box 192, apix 1.383 A, physical field 265.6 A
- each cryoDRGN/subhetero map was pairwise rigid-aligned onto *each* CryoSPARC class; the match is the class with the best FSC-based agreement score (not raw CC).

### Cross-method assignment (best CryoSPARC match)

| map | best match | score | CC(lp) | FSC0.143 (A) | rot (deg) | runner-up (score) |
|---|---|---|---|---|---|---|
| cryodrgn:basinP6 | **P6** | 0.42 | 0.96 | 11.5 | 180 | AltNBD1a (0.32) |
| cryodrgn:basinP7 | **P6** | 0.75 | 0.98 | 5.0 | 12 | P8 (0.38) |
| cryodrgn:basinP8 | **P8** | 0.50 | 0.96 | 9.5 | 180 | P6 (0.43) |
| cryodrgn:basinP9 | **P7** | 0.32 | 0.88 | 19.0 | 180 | P8 (0.24) |
| cryodrgn:basinP10 | **AltNBD1b** | 0.82 | 0.98 | 6.6 | 3 | P7 (0.72) |

### Matched-pair differences (non-resolution structural metrics emphasised)

CC/SSIM/NMI/local-agree: higher = more similar; diff RMS: lower = more similar. FSC columns kept for reference only.

| pair | CC | SSIM | NMI | local-agree median | local-agree p10 | diff RMS | FSC0.143 (A) |
|---|---|---|---|---|---|---|---|
| J1442basin_vs_best_basinP6_vs_P6 | 0.960 | 0.513 | 0.158 | 0.05 | -0.02 | 1.16 | 11.5 |
| J1442basin_vs_best_basinP7_vs_P6 | 0.975 | 0.661 | 0.176 | 0.05 | -0.02 | 0.90 | 5.0 |
| J1442basin_vs_best_basinP8_vs_P8 | 0.959 | 0.514 | 0.155 | 0.06 | -0.03 | 1.11 | 9.5 |
| J1442basin_vs_best_basinP9_vs_P7 | 0.878 | 0.491 | 0.129 | 0.06 | -0.03 | 1.11 | 19.0 |
| J1442basin_vs_best_basinP10_vs_AltNBD1b | 0.980 | 0.750 | 0.148 | 0.05 | -0.02 | 0.79 | 6.6 |
