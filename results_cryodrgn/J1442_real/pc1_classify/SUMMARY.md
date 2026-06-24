# cryoDRGN PC1 classification, overlap, and ab-initio/NU export sets

- particles: 230,396  |  classes (k): 3  |  labels: P6, P7, P8
- PC1 explained variance ratio: 0.155
- PC1-GMM confident (resp >= 0.8): 220,026 (95.5%)

## Class counts

| class | CryoSPARC | PC1 crude | PC1 GMM (all) | PC1 GMM (confident) |
|---|---|---|---|---|
| P6 | 83,843 | 76,799 | 92,696 | 91,885 |
| P7 | 67,671 | 76,798 | 49,715 | 43,040 |
| P8 | 78,882 | 76,799 | 87,985 | 85,101 |

## Particle overlap (adjusted Rand index; 1=identical, 0=chance)

| partitions | ARI | AMI | NMI |
|---|---|---|---|
| cryosparc vs pc1_crude | 0.452 | 0.404 | 0.404 |
| cryosparc vs pc1_gmm | 0.519 | 0.448 | 0.448 |
| cryosparc vs latent_gmm | 0.543 | 0.469 | 0.469 |
| pc1_crude vs pc1_gmm | 0.699 | 0.711 | 0.711 |
| pc1_crude vs latent_gmm | 0.736 | 0.736 | 0.736 |
| pc1_gmm vs latent_gmm | 0.927 | 0.889 | 0.889 |

## The 3 ab-initio -> NU runs (import each .cs, Ab-initio init 12 A / final 4 A, then NU)

1. **RUN 1 - crude PC1 division**:  pc1_crude_P*.cs  (equal-population PC1 tertiles)
2. **RUN 2 - PC1 3-comp GMM, confident only**:  pc1_gmm_conf_P*.cs (also pc1_gmm_P*.cs = all hard-assigned)
3. **RUN 3 - full latent-space GMM**:  cryodrgn_class_P*.cs (from export_cryodrgn_subsets.py)

## .cs files written

- pc1_crude_P6.cs : 76,799 particles
- pc1_crude_P7.cs : 76,798 particles
- pc1_crude_P8.cs : 76,799 particles
- pc1_gmm_P6.cs : 92,696 particles
- pc1_gmm_P7.cs : 49,715 particles
- pc1_gmm_P8.cs : 87,985 particles
- pc1_gmm_conf_P6.cs : 91,885 particles
- pc1_gmm_conf_P7.cs : 43,040 particles
- pc1_gmm_conf_P8.cs : 85,101 particles
