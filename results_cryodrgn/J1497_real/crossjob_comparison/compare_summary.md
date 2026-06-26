# J1442 (3-class) vs J1497 (5-class) cryoDRGN comparison

- particles: 3-class 230,396 | 5-class 230,396 | common 230,396

## A. Latent reproducibility (same images, two runs)
- canonical correlations (top 4): [0.981, 0.981, 0.979, 0.974]
- PC1<->PC1 correlation: 0.898 (PC1 expl.var 3class 0.155 / 5class 0.155)

## B. How many states does the latent support?
- PC1 density modes: 3class=3 | 5class=2
- silhouette at design K: 3class(K=3)=0.095 | 5class(K=5)=0.039
  (silhouette << 0.25 => components overlap; no discrete clusters)

## C. Can the CryoSPARC partition be recovered from the latent? (LDA CV)
- 3-class balanced accuracy 0.799 (chance 0.333, 2.40x lift)
- 5-class balanced accuracy 0.485 (chance 0.200, 2.43x lift)
- 5-class per-class recall: {'P6': 0.901, 'P7': 0.64, 'P8': 0.861, 'P9': 0.016, 'P10': 0.01}

## D. Alternative unsupervised classifiers on the 5-class latent
- KMeans ARI 0.147 / AMI 0.168
- full-cov GMM ARI 0.193 / AMI 0.218
- HDBSCAN ARI 0.000 / AMI 0.000 | found 0 dense clusters (100% noise)

## Verdict
- If PC1 is unimodal, silhouette is low at the design K, supervised
  recovery is only a small multiple of chance, and no unsupervised
  method reproduces the partition, then the extra CryoSPARC classes are
  re-slicing one continuous landscape rather than resolving new states.
