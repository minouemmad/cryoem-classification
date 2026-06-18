# cryoDRGN classification comparison

- latent z: 3000 particles, zdim=4
- clusters: k=2 (KMeans + Gaussian mixture)
- embedding for plots: UMAP

## Agreement metrics (1.0 = identical partition, 0.0 = chance)

| reference | clustering | ARI | AMI | NMI | V-measure |
|---|---|---|---|---|---|
| ground_truth | kmeans | 1.000 | 1.000 | 1.000 | 1.000 |
| ground_truth | gmm | 1.000 | 1.000 | 1.000 | 1.000 |

## Files
- metrics.csv — all scores
- latent_*.png — z embedding coloured by each clustering/reference
- confusion_*.png — Hungarian-aligned confusion matrices
