# cryoDRGN 2D free-energy basins + occupancy matrix

2D free energy `F(PC1,PC2) = -log p` clustered by watershed persistence (a basin survives if its barrier from the shallower side exceeds `--barrier-kt`). The occupancy matrix gives `P(basin | CryoSPARC class)` -- a near-identity matrix means the classes are distinct states; rows sharing a basin reveal classes that are sub-divisions of one free-energy minimum.

## J1442x5 (5-class, 230,396 particles)

- PC1 15.5% / PC2 11.9% of standardized latent variance.
- **2D basins @ 0.5 kT: 3** (raw watershed found 11; tail basins < 1% absorbed). 1-D F(PC1) reported 3 wells for both datasets.
- basin minima: B1=(PC1 -1.26, PC2 0.09), B2=(PC1 0.03, PC2 -0.42), B3=(PC1 1.21, PC2 0.70).
- basin populations: [0.404, 0.25, 0.346].
- top basin persistences (kT): 11.30, 9.15, 1.38, 1.28, 1.25, 1.24.

  Occupancy `P(basin | class)` (hard):

  | class | Basin 1 | Basin 2 | Basin 3 |
  |---|---|---|---|
  | P6 | 0.91 | 0.07 | 0.02 |
  | P7 | 0.18 | 0.69 | 0.13 |
  | P8 | 0.04 | 0.11 | 0.85 |
  | P9 | 0.10 | 0.20 | 0.70 |
  | P10 | 0.65 | 0.26 | 0.09 |


  ### Hybrid-pipeline plan (J1442x5)

  Each basin = one **energetic state** (NU-refine its particle subset). CryoSPARC classes that pile into the *same* basin are candidate **structural substates** of that one energetic state; the within-basin hetero-refine `K` is just how many classes share the basin (`K=1` => the basin already is a single class).

  | basin | population | CryoSPARC classes in basin | within-basin hetero-refine K |
  |---|---|---|---|
  | Basin 1 | 0.404 | P6 (0.91), P10 (0.65) | **K=2** |
  | Basin 2 | 0.250 | P7 (0.69) | K=1 (already one class; no split to test) |
  | Basin 3 | 0.346 | P8 (0.85), P9 (0.70) | **K=2** |

  **Which classes to pair / what K:** Basin 1: heteroref K=2 on P6+P10; Basin 3: heteroref K=2 on P8+P9.
  The pairing is *not* chosen by hand -- it is read off the occupancy matrix: the classes listed in a basin row above are exactly the ones that share that free-energy minimum, so they are the substates to try to re-separate *within that basin only*.

  ### Exact maps to compare

  Independent CryoSPARC refinements are NOT in a common frame -- rigid-body align before every comparison (`scripts/cryodrgn/cryodrgn_focused_analysis.py` aligns class1->class0; for basin maps use ChimeraX fitmap).

  1. **Energetic-state maps (between basins):** NU-refine each `basin_N_particles.cs`, then compare the basin maps pairwise (FSC, masked CC, difference map). These answer *are the free-energy basins genuinely distinct states?*
  2. **Substate maps (within Basin 1):** focused hetero-refine K=2 on `basin1_particles.cs` -> P6-like/P10-like; NU-refine each and compare them to each other (FSC, local resolution, CC, difference map, occupancies). These answer *do P6+P10 survive as substates from images alone?*
  2. **Substate maps (within Basin 3):** focused hetero-refine K=2 on `basin3_particles.cs` -> P8-like/P9-like; NU-refine each and compare them to each other (FSC, local resolution, CC, difference map, occupancies). These answer *do P8+P9 survive as substates from images alone?*
  3. **Cross-pipeline (the headline):** compare each hybrid substate map to the ORIGINAL CryoSPARC hetero-refine K=5 map of the same class (e.g. Basin-1 substate vs original P6 and P10) via a CC matrix. If the hybrid reproduces the 5 original maps, you have shown `5 reconstructable maps = 3 basins x substates` -- structural heterogeneity nested inside fewer energetic states.
