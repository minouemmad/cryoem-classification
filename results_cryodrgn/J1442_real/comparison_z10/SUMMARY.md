# cryoDRGN classification comparison

- latent z: 230396 particles, zdim=10
- clusters: k=3 (KMeans + GMM)
- plot embedding: UMAP

## Agreement (1.0 = identical, 0.0 = chance)

| reference | clustering | ARI | AMI | NMI | V-measure |
|---|---|---|---|---|---|
| cryosparc_class | kmeans | 0.486 | 0.420 | 0.420 | 0.420 |
| cryosparc_class | gmm | 0.543 | 0.469 | 0.469 | 0.469 |
| existing_gmm | kmeans | 0.214 | 0.263 | 0.263 | 0.263 |
| existing_gmm | gmm | 0.235 | 0.304 | 0.304 | 0.304 |
