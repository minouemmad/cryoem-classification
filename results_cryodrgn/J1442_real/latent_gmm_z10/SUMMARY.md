# cryoDRGN latent-space GMM: populations, divergence, failure diagnosis

- particles matched: 230,396  |  zdim: 10  |  K: 3
- BIC-best K (1..8): 8   (dBIC K=3 vs K=1: 191993.4)
- latent-GMM mean max-resp: 0.997  (frac>0.9: 0.991)
- component separation (SD): min 1.90, mean 2.30  |  silhouette 0.077
- canonical corr z vs CryoSPARC: [0.781, 0.348] (max 0.781)
- hard agreement CryoSPARC vs latent-GMM: 0.814
- mean / median JS divergence: 0.298 / 0.299 nats (bound ln2 = 0.693)

## Populations (95% bootstrap CI)

| class | CryoSPARC soft | latent-GMM soft | latent-GMM corrected |
|---|---|---|---|
| P6 | 0.331 | 0.402 [0.400, 0.404] | 0.365 [0.363, 0.367] |
| P7 | 0.337 | 0.235 [0.233, 0.236] | 0.293 [0.291, 0.295] |
| P8 | 0.332 | 0.363 [0.361, 0.365] | 0.343 [0.341, 0.345] |

## Failure-mode verdict

- min component separation 1.90 SD (<2) -> latent GMM components are arbitrary slices of one cloud.

## Suggested next steps

- If the latent is unimodal / low canonical-corr: this corroborates the flat-posterior finding — report a continuous reaction coordinate (pc1/UMAP traversal) and populations along it, not discrete fractions.
- To stress-test cryoDRGN itself before concluding: retrain at higher `--zdim` (e.g. 10) and/or more epochs, and confirm the z.N.pkl learning curve plateaued; re-run this script and check whether ΔBIC and canonical correlation improve. If they don't, the continuity is data-driven, not an undertraining artifact.
- For a discrete read if warranted: feed `cryodrgn_compare.py` for the ARI/NMI clustering view, and use the per_particle.npz JS/agreement to select a high-confidence (low-JS, agree==1) subset for refinement.
