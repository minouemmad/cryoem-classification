# Legacy

Earlier approaches kept for reference; not part of the current pipeline (see the top-level README):
reconstruction-guided latent trees (`recon_tree.py`, `dataset/run_pipeline.sh`, leaf chains), latent-space and decoder diagnostics,
focused cryoDRGN, PC1 trajectories, and prior / deformation variants of the trainer (`train_vae_gmmprior.py`, `train_vae_diffprior.py`,
`train_rae_fprior.py`, `train_rae_deform.py`). Paths inside these scripts point at the old `scripts/cryodrgn/` layout.

`window_methods/` (from the 2026-11-23 layout): P7-window / PC1-end-ladder volume finding, the first residual-axis split, map scoring
against the J2923 classes (own-state score, size plateau, off-line rule) and the lanes that ran them. Superseded by pipeline/4_residual - 6_classes.
