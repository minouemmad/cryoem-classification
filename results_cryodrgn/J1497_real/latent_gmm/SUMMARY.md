# cryoDRGN latent-space GMM: populations, component separation, and cross-method alignment

- particles matched: 230,396  |  zdim: 10  |  K: 5
- BIC-best K (1..10): 10   (dBIC K=5 vs K=1: 268202.5)
- latent-GMM mean max-resp: 0.902  (frac>0.9: 0.683)
- component separation (SD): min 0.88, mean 1.95  |  silhouette 0.044
- canonical corr z vs CryoSPARC: [0.761, 0.288, 0.144, 0.035] (max 0.761)
- hard agreement CryoSPARC vs latent-GMM: 0.561
- mean / median JS divergence: 0.348 / 0.372 nats (bound ln2 = 0.693)

## Populations (95% bootstrap CI)

| class | CryoSPARC soft | latent-GMM soft | latent-GMM corrected |
|---|---|---|---|
| P6 | 0.199 | 0.380 [0.378, 0.382] | 0.226 [0.223, 0.229] |
| P7 | 0.203 | 0.225 [0.224, 0.227] | 0.172 [0.170, 0.175] |
| P8 | 0.200 | 0.148 [0.146, 0.149] | 0.313 [0.309, 0.316] |
| P9 | 0.199 | 0.208 [0.206, 0.209] | 0.046 [0.043, 0.049] |
| P10 | 0.198 | 0.039 [0.039, 0.040] | 0.243 [0.238, 0.247] |

## Evaluation

- min component separation 0.88 SD: the 5 GMM components heavily overlap; K=5 is not supported by the latent density.
- BIC is monotonically decreasing through K=10 (no elbow); the 5-class partition has no intrinsic support in the latent.
- Hard agreement 56.1% (vs 81.4% for J1442 K=3): the 5-class partition is substantially harder for cryoDRGN to reproduce.
- LDA supervised recall: P6=0.90, P7=0.64, P8=0.86, P9=0.016, P10=0.010; P9 and P10 are structurally indistinguishable from P8 and P6, respectively.
- See `crossjob_comparison/` for the full cross-job CCA and alternative clustering analysis.
