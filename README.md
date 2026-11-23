# cryoem-classification

Label-free discovery of conformational states in hCFTR cryo-EM particle sets. A cryoDRGN-style autoencoder is trained on
consensus poses, its latent space is reduced to a main axis (PC1), and candidate particle groups are chosen along that axis
without assuming the number or size of states. Each group is refined in cryoSPARC (ab initio + non-uniform refinement) and
the resulting maps are scored for reproducibility and for how distinct they are from the end states and from a random mixture.
Heterogeneous-refinement classes are used only to evaluate results, never to pick groups.

## Pipeline

| Stage | Folder | What it does |
|---|---|---|
| 0 | `scripts/00_dataset_setup/` | Import a cryoSPARC job as a dataset: NU poses, CTF, 128 px stack, mask |
| 1 | `scripts/01_train/` | Train the autoencoder (RAE with gradient penalty; VAE and variants for comparison) |
| 2 | `scripts/02_decode_and_pc/` | Decode ~2,000 volumes; per-particle volume-PC scores (`volume_pc_scores.py`) or latent PC1 (`latent_pc1.py`) |
| 3 | `scripts/03_find_volumes/` | Choose particle groups (see below) |
| 4 | `scripts/04_refine_cryosparc/` | Export groups as STAR files, create/queue cryoSPARC chains (ab initio -> NU) |
| 5 | `scripts/05_score_maps/` | Align and score maps, off-line rule, own-state score, state mixture fit, size plateau, reports |
| - | `scripts/lanes/` | GPU lane scripts chaining train -> decode -> scores -> residual split -> refinement |
| - | `scripts/lib/` | Shared modules (`cryo_utils`, `fastbp`, `mapdiff`, `recon_tree`) |
| - | `scripts/tools/`, `scripts/trajectory/` | Map comparison, focused cryoDRGN, latent clustering, PC1 trajectories |

### Stage 3 - ways to find volumes
- `pc1_windows/` - sliding windows along PC1: hard, Gaussian soft, fixed PC value width, equal arc length, kNN neighbourhoods, replicate draws.
- `pc1_ends/` - nested windows from each end of PC1 (end ladder); the end map is the largest window still within seed noise of its core.
- `residual_axis/` - fit residual axes (`residual_fit.py`) on the middle of PC1 after removing the PC1 average map, then split the middle window along them (`residual_split.py`, `build_axis_cut.py`).
- `lib/recon_tree.py` - reconstruction-guided latent tree (earlier approach).

## Findings so far
- The two PC1 end states are reproducible across seeds, window sizes and models.
- PC1 windows give a reproducible but mixed intermediate map; the residual-axis split enriches it further (not pure).
- No latent axis isolated the small alternative-NBD1 state.
- Whether a dataset's latent captures heterogeneity must be checked first (J207's RAE/VAE latent does not separate its two heteroref classes).

## Running
Use the `cryodrgn` conda env. Scripts carry cluster paths (`/data/hCFTR/...`, cryoSPARC project/job ids) at the top; adjust them
for another machine. Data and results are not versioned (see `data/README.md`). Datasets: `J4001_gP25_WT`, `J4624_eP37_WT_T2a`,
`J399_hP7_E1371Q_ATP`, `J2708_eP30_F508del_E1371W`, `J207_hP12_IDOR4`.
