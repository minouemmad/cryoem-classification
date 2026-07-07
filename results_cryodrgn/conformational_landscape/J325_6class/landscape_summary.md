# Conformational-landscape analysis

- Particles: 300,437; latent dim 10
- PCA variance PC1/2/3 = 14.6% / 11.0% / 10.8%

## Discrete or continuous? (free-energy wells per PC)
- F(PC1): 1 well(s); barriers [] kT
- F(PC2): 1 well(s); barriers [] kT
- F(PC3): 1 well(s); barriers [] kT

## Class self-consistency & most-confused partner (k-NN local / GMM global)
- SC: self kNN 0.777 / GMM 0.862, nearest other = **AC** (0.064)
- AC: self kNN 0.697 / GMM 0.777, nearest other = **SEPD** (0.081)
- AO: self kNN 0.603 / GMM 0.728, nearest other = **SEPD** (0.154)
- SEPD: self kNN 0.576 / GMM 0.591, nearest other = **AO** (0.114)
- AEPD: self kNN 0.543 / GMM 0.577, nearest other = **V-shaped** (0.129)
- V-shaped: self kNN 0.21 / GMM 0.197, nearest other = **AEPD** (0.329)

Class map: P6=SC, P7=AC, P8=AO, P9=SEPD, P10=AEPD, P11=V-shaped
