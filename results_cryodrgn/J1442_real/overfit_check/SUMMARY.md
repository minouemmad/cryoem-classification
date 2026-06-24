# cryoDRGN overfitting / over-confidence diagnostics

- particles: 230,396  |  zdim: 10  |  k: 3
- PC1 explained variance ratio: 0.155

## Verdict

- OVER-CONFIDENT: latent-GMM assigns 0.997 mean confidence though components are only 1.90 SD apart (expected ~0.829); gap 0.168. The discrete K-state model manufactures confidence on a continuous cloud - this is the 'too confident PC1' effect, NOT evidence of real wells.
- PC1 is largely free of imaging confounds (max |r|=0.02, R^2=0.00) -> PC1 reflects structure, not CTF/pose/scale.
- STABLE: PC1 locks to its final direction by epoch 73 (|corr|>=0.95) and holds -> converged coordinate, not late-training memorisation.
- Train loss converged (final 0.5588, tail-20 relative change 0.01%). NOTE: cryoDRGN train_vae holds out no validation set, so flat train loss alone cannot exclude overfitting - checks (1)-(3) are the decisive ones.

## Key numbers

- mean max responsibility: 0.997 (frac>0.9 = 0.991)
- min component separation: 1.90 SD (expected max-resp at this separation: 0.829)
- over-confidence gap: 0.168
- PC1 max |confound corr|: 0.02 (particle_scale_alpha); multivariate R^2 0.00
- PC1 lock-in epoch (|corr|>=0.95): 73; final-epoch corr 1.000
