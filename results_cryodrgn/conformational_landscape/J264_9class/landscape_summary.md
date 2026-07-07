# Conformational-landscape analysis

- Particles: 299,745; latent dim 10
- PCA variance PC1/2/3 = 14.6% / 11.0% / 10.9%

## Discrete or continuous? (free-energy wells per PC)
- F(PC1): 1 well(s); barriers [] kT
- F(PC2): 1 well(s); barriers [] kT
- F(PC3): 1 well(s); barriers [] kT

## Class self-consistency & most-confused partner (k-NN local / GMM global)
- SC: self kNN 0.924 / GMM 0.979, nearest other = **AEPD** (0.042)
- AC: self kNN 0.939 / GMM 0.97, nearest other = **V-shaped** (0.019)
- AO: self kNN 0.841 / GMM 0.957, nearest other = **SEPD** (0.113)
- SEPD: self kNN 0.72 / GMM 0.871, nearest other = **AO** (0.078)
- AEPD: self kNN 0.654 / GMM 0.852, nearest other = **V-shaped** (0.141)
- V-shaped: self kNN 0.275 / GMM 0.57, nearest other = **AEPD** (0.374)
- NBD1-less: self kNN 0.188 / GMM 0.355, nearest other = **SEPD** (0.41)
- NBD2-less: self kNN 0.235 / GMM 0.481, nearest other = **SEPD** (0.358)
- NBD1-less-wide: self kNN 0.029 / GMM 0.073, nearest other = **AEPD** (0.333)

## Core states (ablated excluded)
- kept: SC, AC, AO, SEPD, AEPD, V-shaped
- PC1'/2'/3' var: [0.133, 0.117, 0.111]

Class map: P6=SC, P7=AC, P8=AO, P9=SEPD, P10=AEPD, P11=V-shaped, P12=NBD1-less, P13=NBD2-less, P14=NBD1-less-wide
