# cryoDRGN overfitting / over-confidence diagnostics

- particles: 230,396  |  zdim: 10  |  k: 5
- PC1 explained variance ratio: 0.155

## Verdict

- OVER-CONFIDENT: latent-GMM assigns 0.902 mean confidence though components are only 0.88 SD apart (expected ~0.671); gap 0.231. The discrete K-state model manufactures confidence on a continuous cloud - this is the 'too confident PC1' effect, NOT evidence of real wells.
- PC1 is largely free of imaging confounds (max |r|=0.02, R^2=0.00) -> PC1 reflects structure, not CTF/pose/scale.
- STABLE: PC1 locks to its final direction by epoch 82 (|corr|>=0.95) and holds -> converged coordinate, not late-training memorisation.
- Train loss converged (final 0.5588, tail-20 relative change 0.01%). NOTE: cryoDRGN train_vae holds out no validation set, so flat train loss alone cannot exclude overfitting - checks (1)-(3) are the decisive ones.

## Key numbers

- mean max responsibility: 0.902 (frac>0.9 = 0.683)
- min component separation: 0.88 SD (expected max-resp at this separation: 0.671)
- over-confidence gap: 0.231
- PC1 max |confound corr|: 0.02 (df_angle_rad); multivariate R^2 0.00
- PC1 lock-in epoch (|corr|>=0.95): 82; final-epoch corr 1.000
