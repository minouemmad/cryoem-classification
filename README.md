# CFTR Cryo-EM Classification & Conformational Landscape Pipeline

Combines **CryoSPARC heterogeneous-refinement uncertainty quantification** (GMM pipeline) with **cryoDRGN neural-network conformational-landscape analysis** to characterise drug-bound CFTR conformations.

**Core scientific result:** Two independent methods (CryoSPARC + cryoDRGN D=256 converged) both recover the same **3 core conformational states** of CFTR+Trikafta. CryoSPARC posteriors are near-uniform after debiasing (mean max-posterior ~0.36, genuinely ambiguous), while cryoDRGN's 10-D latent GMM separates the 3 states at **2.60 SD** (>2 = discrete).

## Repository layout

```
run_pipeline.py            GMM pipeline: .cs -> ALR -> GMM -> confusion -> populations -> exports
scripts/
  gmm_pipeline/            Core library: data_io, preprocess, gmm_fit, confusion, uncertainty, plots
  pipeline/                Particle exports, GMM bootstrap, ensembles, re-plotting
  maps/                    Map comparison (CC/FSC), ChimeraX alignment, density diagnostics, 3DVA
  cryodrgn/                cryoDRGN analysis: landscape, basin occupancy, free energy, cluster export
    class_names.py           Biological class labels (J1442/J1497/J264/J325) -- single source of truth
  diagnostics/             Posterior sanity plots, uncertainty comparisons, pairwise scatter
data/                      CryoSPARC .cs inputs by job ID
results_cryodrgn/
  J1442/                   K=3, 230,396 particles
    fullset_D256_z10_ep100/  CONVERGED D=256 run; K=3 GMM min sep 2.60 SD; use this
      latent_gmm_k3/          GMM assignments; cluster_exports/ has .cs per component
      landscape_k3/           PCA landscape; panel_D shows 3 PC1 peaks
      landscape_k5/           K=5 (min sep 0.79 SD -- over-partitioned)
    fullset_D128_z10_ep100/  D=128 original run (landscape_z10 shows 3-modal PC1)
    purified_D256_z10_ep50/  Purified subset (ind), 50 epochs
    confidence_3class/       CryoSPARC vs cryoDRGN cross-method confusion (K=3)
    confidence_5class/       K=5 confusion; P9->P8 44%, P10->P6 49% (sub-states)
  J264/                    K=9, 301,770 particles
    fullset_D256_z10_ep50/   CONVERGED D=256; F(PC1)=1 continuous basin; all 9 classes B1=1.00
      landscape_k9/           Class-labelled landscape with bio names
      free_energy/            Free energy figure: single well, CONTINUOUS
      basin_occupancy/        2D watershed: 1 basin
    purified_D256_z10_ep75/  Purified subset, 75 epochs
results_cryosparc/         CryoSPARC analysis, map comparisons, diagnostics
docs/
  WORKFLOW.md                End-to-end analysis workflow
  CFTR_cryoDRGN_presentation.pptx  15-slide presentation (bio background -> methods -> results)
```

## Biological class names (see scripts/cryodrgn/class_names.py)

| Dataset | Index | Name |
|---------|-------|------|
| J1442/J1497 | P6 | NBD1LessMix-Ablated |
| J1442/J1497 | P7 | NBD1LessWide-Ablated |
| J1442/J1497 | P8 | VshapedMix |
| J1442/J1497 | P9 | NBD2Less-Ablated |
| J1442/J1497 | P10 | AltNBD1-ArdeconComposite-Ablated |
| J264/J325 | P6-P14 | SC, AC, AO, SEPD, AEPD, V-shaped, NBD1-less, NBD2-less, NBD1-less-wide |

## Quick start

```powershell
# GMM pipeline (no GPU)
python run_pipeline.py --cs data/cryosparc_P25_J1442_00000_particles.cs --n-dummies 6 --outdir results_J1442

# cryoDRGN landscape from existing latents (no GPU, no PYTHONPATH needed)
cryodrgn-py310\Scripts\python.exe scripts\cryodrgn\cryodrgn_landscape.py `
    --z results_cryodrgn\J1442\fullset_D256_z10_ep100\z.100.pkl `
    --passthrough-cs data\cryosparc_P25_J1442_passthrough_particles_all_classes.cs `
    --cs data\cryosparc_P25_J1442_00000_particles.cs `
    --n-dummies 6 --protein-idx 6 7 8 -k 3 --dataset J1442 `
    -o results_cryodrgn\J1442\fullset_D256_z10_ep100\landscape_k3
```

See [docs/WORKFLOW.md](docs/WORKFLOW.md) for the full workflow.
