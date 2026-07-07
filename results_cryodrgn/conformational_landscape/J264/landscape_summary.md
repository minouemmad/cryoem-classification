# Conformational-landscape analysis

- Particles: 299,745; latent dim 10
- PCA variance PC1/2/3 = 14.6% / 11.0% / 10.9%

## Discrete or continuous? (free-energy wells per PC)
- F(PC1): 1 well(s); barriers [] kT
- F(PC2): 1 well(s); barriers [] kT
- F(PC3): 1 well(s); barriers [] kT

## Most-confused class partner (k-NN latent neighbours)
- SC: self 0.924, nearest other = **AEPD** (0.042)
- AC: self 0.939, nearest other = **V-shaped** (0.019)
- AO: self 0.841, nearest other = **SEPD** (0.113)
- SEPD: self 0.72, nearest other = **AO** (0.078)
- AEPD: self 0.654, nearest other = **V-shaped** (0.141)
- V-shaped: self 0.275, nearest other = **AEPD** (0.374)
- NBD1-less: self 0.188, nearest other = **SEPD** (0.41)
- NBD2-less: self 0.235, nearest other = **SEPD** (0.358)
- NBD1-less-wide: self 0.029, nearest other = **AEPD** (0.333)

## Core states (ablated excluded)
- kept: SC, AC, AO, SEPD, AEPD, V-shaped
- PC1'/2'/3' var: [0.133, 0.117, 0.111]

Class map: P6=SC, P7=AC, P8=AO, P9=SEPD, P10=AEPD, P11=V-shaped, P12=NBD1-less, P13=NBD2-less, P14=NBD1-less-wide
