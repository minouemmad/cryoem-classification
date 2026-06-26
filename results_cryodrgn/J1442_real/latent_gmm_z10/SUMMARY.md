# cryoDRGN latent-space GMM: populations, component separation, and cross-method alignment

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

## Evaluation

- min component separation 1.90 SD (<2): components are near the discrete-well threshold but remain below it; the latent density is continuous.
- BIC-best K is 8, not 3: the model prefers more components to describe the same cloud; no objective reason to prefer K=3 over other choices.
- Despite continuity, canonical corr = 0.781 confirms the latent encodes the same conformational axis as CryoSPARC.
- See `landscape_z10/` for PCA/UMAP visualization and `crossjob_comparison/` for cross-job CCA and LDA recoverability.
