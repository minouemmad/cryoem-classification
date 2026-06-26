# cryoDRGN PC1 classification, overlap, and ab-initio/NU export sets

- particles: 230,396  |  classes (k): 5  |  labels: P6, P7, P8, P9, P10
- PC1 explained variance ratio: 0.155
- PC1-GMM confident (resp >= 0.8): 147,849 (64.2%)

## Class counts

| class | CryoSPARC | PC1 crude | PC1 GMM (all) | PC1 GMM (confident) |
|---|---|---|---|---|
| P6 | 66,134 | 46,078 | 90,522 | 85,354 |
| P7 | 48,318 | 46,080 | 49,607 | 35,741 |
| P8 | 53,397 | 46,079 | 72,444 | 20,352 |
| P9 | 34,075 | 46,079 | 16,100 | 5,147 |
| P10 | 28,472 | 46,080 | 1,723 | 1,255 |

## Particle overlap (adjusted Rand index; 1=identical, 0=chance)

| partitions | ARI | AMI | NMI |
|---|---|---|---|
| cryosparc vs pc1_crude | 0.246 | 0.279 | 0.279 |
| cryosparc vs pc1_gmm | 0.323 | 0.312 | 0.312 |
| cryosparc vs latent_gmm | 0.345 | 0.324 | 0.324 |
| pc1_crude vs pc1_gmm | 0.586 | 0.736 | 0.736 |
| pc1_crude vs latent_gmm | 0.516 | 0.615 | 0.615 |
| pc1_gmm vs latent_gmm | 0.766 | 0.710 | 0.710 |

## The 3 ab-initio -> NU runs (import each .cs, Ab-initio init 12 A / final 4 A, then NU)

1. **RUN 1 - crude PC1 division**:  pc1_crude_P*.cs  (equal-population PC1 tertiles)
2. **RUN 2 - PC1 3-comp GMM, confident only**:  pc1_gmm_conf_P*.cs (also pc1_gmm_P*.cs = all hard-assigned)
3. **RUN 3 - full latent-space GMM**:  cryodrgn_class_P*.cs (from export_cryodrgn_subsets.py)

## .cs files written

- pc1_crude_P6.cs : 46,078 particles
- pc1_crude_P7.cs : 46,080 particles
- pc1_crude_P8.cs : 46,079 particles
- pc1_crude_P9.cs : 46,079 particles
- pc1_crude_P10.cs : 46,080 particles
- pc1_gmm_P6.cs : 90,522 particles
- pc1_gmm_P7.cs : 49,607 particles
- pc1_gmm_P8.cs : 72,444 particles
- pc1_gmm_P9.cs : 16,100 particles
- pc1_gmm_P10.cs : 1,723 particles
- pc1_gmm_conf_P6.cs : 85,354 particles
- pc1_gmm_conf_P7.cs : 35,741 particles
- pc1_gmm_conf_P8.cs : 20,352 particles
- pc1_gmm_conf_P9.cs : 5,147 particles
- pc1_gmm_conf_P10.cs : 1,255 particles
