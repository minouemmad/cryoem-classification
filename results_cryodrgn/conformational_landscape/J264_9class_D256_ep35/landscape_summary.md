# Conformational-landscape analysis

- Particles: 218,918; latent dim 10
- PCA variance PC1/2/3 = 23.3% / 16.3% / 13.3%

## Discrete or continuous? (free-energy wells per PC)
- F(PC1): 1 well(s); barriers [] kT
- F(PC2): 1 well(s); barriers [] kT
- F(PC3): 1 well(s); barriers [] kT

## Class self-consistency & most-confused partner (k-NN local / GMM global)
- SC: self kNN 0.967 / GMM 0.987, nearest other = **AC** (0.016)
- AC: self kNN 0.968 / GMM 0.977, nearest other = **SC** (0.012)
- AO: self kNN 0.064 / GMM 0.182, nearest other = **SEPD** (0.431)
- SEPD: self kNN 0.91 / GMM 0.964, nearest other = **AC** (0.05)
- AEPD: self kNN 0.954 / GMM 0.971, nearest other = **V-shaped** (0.021)
- V-shaped: self kNN 0.808 / GMM 0.963, nearest other = **AEPD** (0.122)
- NBD1-less: self kNN 0.747 / GMM 0.851, nearest other = **SEPD** (0.076)
- NBD2-less: self kNN 0.738 / GMM 0.887, nearest other = **V-shaped** (0.082)
- NBD1-less-wide: self kNN 0.3 / GMM 0.701, nearest other = **AEPD** (0.24)

## Core states (ablated excluded)
- kept: SC, AC, AO, SEPD, AEPD, V-shaped
- PC1'/2'/3' var: [0.224, 0.171, 0.138]

Class map: P6=SC, P7=AC, P8=AO, P9=SEPD, P10=AEPD, P11=V-shaped, P12=NBD1-less, P13=NBD2-less, P14=NBD1-less-wide
