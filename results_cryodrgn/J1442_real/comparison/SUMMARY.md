# cryoDRGN classification comparison

- latent z: 230396 particles, zdim=8
- clusters: k=3 (KMeans + GMM)
- plot embedding: UMAP

## Agreement (1.0 = identical, 0.0 = chance)

| reference | clustering | ARI | AMI | NMI | V-measure |
|---|---|---|---|---|---|
| cryosparc_class | kmeans | 0.387 | 0.350 | 0.350 | 0.350 |
| cryosparc_class | gmm | 0.538 | 0.464 | 0.464 | 0.464 |
| existing_gmm | kmeans | 0.163 | 0.194 | 0.194 | 0.194 |
| existing_gmm | gmm | 0.232 | 0.301 | 0.301 | 0.301 |
