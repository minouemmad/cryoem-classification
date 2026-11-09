# cryoem-classification

Label-free conformational classification of hCFTR cryo-EM particles with cryoDRGN: train a VAE (or a prior / autoencoder
variant) on consensus poses, decode representative latent points, and split particles with a reconstruction-guided tree
whose splits must hold up on held-out images.

## Layout

```
data/                cryoSPARC exports per dataset (not versioned; see data/README.md)
results_cryodrgn/    one folder per dataset: inputs, training runs, decoded centres, trees (not versioned)
scripts/cryodrgn/    all code (below)
```

Datasets (named `<heteroref job>_<project>_<construct>`): `J4001_gP25_WT`, `J399_hP7_E1371Q_ATP`, `J207_hP12_IDOR4`,
`J2708_eP30_F508del_E1371W`, `J4624_eP37_WT_T2a`.

## Scripts (`scripts/cryodrgn/`)

| Step | Script |
|---|---|
| Dataset setup (cryoSPARC poses, inputs, VAE, decode, trees) | `dataset/run_pipeline.sh` (+ `prepare.py`, `write_star.py`, `cs_pose_jobs.py`, `reorder_poses.py`, `make_mask.py`, `registry.py`) |
| Extra tree on a finished dataset | `dataset/run_tree.sh` |
| Leaf maps: per-leaf STAR, cryoSPARC ab initio + NU | `dataset/write_star.py NAME TREE TAG`, `dataset/cs_leaf_jobs.py`, `dataset/leaf_chain.sh` |
| Training variants of `cryodrgn train_vae` | `train_vae_gmmprior.py`, `train_vae_diffprior.py`, `train_rae.py`, `train_rae_fprior.py`, `train_rae_deform.py`, `train_ae.py` |
| Decode representative latent points | `decode.py` |
| Reconstruction-guided tree | `recon_tree.py` (uses `fastbp.py`) |
| Trajectories along latent / volume PC1 | `trajectory/` (`prepare.py`, `run.sh`, `fig.py`) |
| Map comparison | `mapdiff.py`, `cx_pairfit.py`, `cx_render_mapdiff.py`, `group_resolution.py` |
| Focused cryoDRGN | `focus_mask.py`, `focus_subtract.py` |
| Latent-space diagnostics | `latent_clustering.py`, `latent_compare.py`, `latent_curvature.py`, `latent_diffusion.py` |
| Decoder diagnostics | `pe_tests.py`, `pe_eval_data.py`, `vae_perturb_vs_data.py` |
| Utilities | `cryo_utils.py` (shared helpers), `cs2star.py`, `topaz_denoise.py` |

Run everything with the `cryodrgn` conda env (`~/miniconda3/envs/cryodrgn/bin/python`); `topaz_denoise.py` needs the
`topaz` env and `cx_*.py` run inside ChimeraX (`chimerax --nogui --offscreen --exit --script ...`).

## New dataset

```bash
# 1. add it to scripts/cryodrgn/dataset/registry.py, then
python scripts/cryodrgn/dataset/prepare.py NAME
python scripts/cryodrgn/dataset/write_star.py NAME
cryosparcm cli "(lambda g: (exec(open('scripts/cryodrgn/dataset/cs_pose_jobs.py').read(), g), g['RESULT'])[1])({'api': api, 'NAME': 'NAME', 'PROJECT': 'P?'})"
# 2. everything else (cryoSPARC chain, cryoDRGN inputs, VAE, decode, trees)
nohup scripts/cryodrgn/dataset/run_pipeline.sh NAME CRYOSPARC_GPU &
```
