# cryoDRGN latent-space GMM: populations, component separation, and cross-method alignment

- particles matched: 230,396  |  zdim: 8  |  K: 3
- BIC-best K (1..8): 7   (dBIC K=3 vs K=1: 286674.2)
- latent-GMM mean max-resp: 0.998  (frac>0.9: 0.994)
- component separation (SD): min 1.34, mean 1.76  |  silhouette 0.054
- canonical corr z vs CryoSPARC: [0.79, 0.208] (max 0.790)
- hard agreement CryoSPARC vs latent-GMM: 0.811
- mean / median JS divergence: 0.300 / 0.301 nats (bound ln2 = 0.693)

## Populations (95% bootstrap CI)

| class | CryoSPARC soft | latent-GMM soft | latent-GMM corrected |
|---|---|---|---|
| P6 | 0.331 | 0.402 [0.400, 0.403] | 0.364 [0.362, 0.366] |
| P7 | 0.337 | 0.236 [0.234, 0.237] | 0.294 [0.292, 0.295] |
| P8 | 0.332 | 0.363 [0.361, 0.364] | 0.342 [0.340, 0.344] |

## Evaluation

- min component separation 1.34 SD (<2): latent GMM components are slicing a continuous distribution, not resolving discrete wells.
- The z10 run (`latent_gmm_z10/`) shows improved separation (1.90 SD) with higher zdim/longer training, confirming continuity is data-driven rather than an undertraining artifact.
- See `crossjob_comparison/` for supervised LDA recoverability and alternative clustering ARI.
