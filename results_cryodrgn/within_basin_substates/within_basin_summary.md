# Within-basin substates + hierarchical uncertainty

Two model-free Stage-2 tests (Option 3 local density peaks, Option 4 diffusion map) run on each cryoDRGN latent, plus Level-1 (basin population) and Level-2 (substate assignment) uncertainty.

## J1442  (230,396 particles, zdim 10)

- **Stage 1 basins:** 3 (populations B1=0.40±0.00, B2=0.25±0.00, B3=0.35±0.00)
- **Diffusion-map spectral gap:** ~1 metastable state(s); |corr(DC1,PC1)| = 1.00
- **Within-basin substates (Option 3):**

  | basin | particles | sub-wells | deepest sub-barrier (kT) | split-half reproducible |
  |---|---|---|---|---|
  | 1 | 93,024 | 1 | 0.00 | False (cos=0.9977139687038449, corr=0.857833880792506) |
  | 2 | 57,690 | 2 | 1.79 | False (cos=0.996973736677339, corr=0.42450993147314814) |
  | 3 | 79,682 | 1 | 0.00 | False (cos=0.6261931902411106, corr=0.845215733188096) |

## J1497  (230,396 particles, zdim 10)

- **Stage 1 basins:** 2 (populations B1=0.43±0.00, B2=0.57±0.00)
- **Diffusion-map spectral gap:** ~1 metastable state(s); |corr(DC1,PC1)| = 1.00
- **Within-basin substates (Option 3):**

  | basin | particles | sub-wells | deepest sub-barrier (kT) | split-half reproducible |
  |---|---|---|---|---|
  | 1 | 98,794 | 1 | 0.00 | False (cos=0.9996737174800213, corr=0.8844389196029727) |
  | 2 | 131,602 | 1 | 0.00 | False (cos=0.9965282476631804, corr=0.9889635389447626) |

## J264  (299,745 particles, zdim 10)

- **Stage 1 basins:** 1 (populations B1=1.00±0.00)
- **Diffusion-map spectral gap:** ~1 metastable state(s); |corr(DC1,PC1)| = 1.00
- **Within-basin substates (Option 3):**

  | basin | particles | sub-wells | deepest sub-barrier (kT) | split-half reproducible |
  |---|---|---|---|---|
  | 1 | 299,745 | 1 | 0.00 | False (cos=0.9996339720329299, corr=0.959537281197754) |


### How to read this
- A basin with **1 sub-well** = one structural blob = no substate to chase (Stage 2 stops).
- A basin with **>=2 reproducible sub-wells** (split-half agrees, high axis cos + curve corr) = a real candidate substate split -> run a **focused hetero-refine K = #sub-wells** on that basin's exported particles.
- Diffusion spectral gap is an *independent* count of metastable states; if it matches the basin count, the Stage-1 picture is robust.
