# cryoem-classification

Label-free conformational classification of hCFTR cryo-EM particles.

The core is a **residual model fitted directly to the images**: a probabilistic PCA of the whitened residuals between each image and the projection of a reference map, at fixed consensus NU poses. A starting axis orders the particles. A conditioned search then adds coordinates one at a time, each fitted on what the earlier ones leave unexplained. Classes are cut in that coordinate space and refined in cryoSPARC (ab initio + NU).

- **Measured on maps, not particles.** Every decision is made on back-projected half maps of the real images, so the quantities used are reproducible between independent halves.
- **Several starting axes.** The start can come from the residual model alone (`free`) or from three nonlinear summaries of it: a codebook (`vq`), a diffusion map (`diffmap`) or an MLP bottleneck (`mlp`). A trained cryoDRGN model (`latent`, or the original volume-PC route) is optional.
- **Reference classes for evaluation only.** cryoSPARC heterorefinement classes are refined by the same route and compared at the end; they never feed a fit, a mask or a choice.

## Layout

```
config/      env.sh (site paths), datasets.tsv (stack, poses, CTF, envelope, mask, voxel size, max rounds), heteroref.json, examples
lib/         fastbp.py (GPU back-projection, half maps, simulation), maps.py, mapdiff.py, cryo_utils.py
pipeline/
  0_consensus/   consensus NU poses -> 128 px stack, ctf.pkl, poses.pkl, STAR
  1_envelope/    protein envelope from a fitted PDB (6MSM / 5UAK) + domain regions
  2_train/       cryoDRGN RAE / VAE / AE / conditional RAE trainers (optional start only)
  3_scores/      start.py: starting axes (rand, free, vq, diffmap, mlp, latent); decode.py + scores.py: cryoDRGN volume-PC start
  4_residual/    residual_ppca.py (image-domain PPCA), sharpen.py;  domains/: per-domain and per-subset residual fits
  5_search/      search.py: fully conditioned search for further coordinates
  6_classes/     classes.py, kselect.py, repro.py, reassign.py;  windows/: variable-size soft windows along any axis
  7_refine/      STAR export, cryoSPARC chains (import -> ab initio -> NU);  remote/: multi-machine dispatcher (cryoSPARC v4 / v5 via cli)
  8_compare/     alignment to a common frame, cc and distance to the heterorefinement maps (evaluation only)
  run_model.sh   steps 3-6 for one run;  pipe_worker.sh (queue);  auto_select.sh (K scan -> refinement)
diagnostics/   evaluation-only checks: reproducibility, start-vs-start agreement, envelope / micelle, P7, domain density
tools/deck/    pptx of every refined map with its nearest heterorefinement class (rebuilt as maps arrive)
reference/     PDB models, envelopes, cryoSPARC job ids, K-scan results (reference/README.md)
legacy/        earlier approaches; not used
```

Datasets: `J4001_gP25_WT`, `J4624_eP37_WT_T2a`, `J207_hP12_IDOR4`, `J2708_eP30_F508del_E1371W` and `J399_hP7_E1371Q_ATP`.

Large data stays on disk (`$RES`, `$WORK` in `config/env.sh`). Python runs in the `cryodrgn` conda env. `cx_*` / `fit_pdb.py` / `fit_union.py` run in headless ChimeraX.

## Running

```bash
source config/env.sh
# 0-1  consensus inputs and envelope (once per dataset; add a row to config/datasets.tsv)
python pipeline/0_consensus/prepare.py NAME && pipeline/0_consensus/consensus.sh NAME GPU
pipeline/1_envelope/envelope.sh GPU DS STACK POSES CTF
# 3-6  one run per start (no training needed for free / vq / diffmap / mlp)
pipeline/run_model.sh GPU DS start_free  - - MP_START=free
pipeline/run_model.sh GPU DS start_vq    - - MP_START=vq,MP_REUSE_START=DS/start_free      # reuses the free-stage residual fit
pipeline/run_model.sh GPU DS start_mlp   - - MP_START=mlp,MP_REUSE_START=DS/start_free,MP_SEED=1
pipeline/run_model.sh GPU DS start_latent $WORK/models/DS/rae_base 50 MP_START=latent       # optional cryoDRGN start
# 6-7  K scan and refinement (one machine: 7_refine/submit.sh; several: 7_refine/remote/dispatch.py)
pipeline/auto_select.sh "DS/start_free:auto";  pipeline/7_refine/submit.sh DS hetero
# windows / domain fits (optional, J4001 defaults)
python pipeline/6_classes/windows/win_build.py DS RUN j stage1;  python pipeline/6_classes/windows/win_score.py
pipeline/4_residual/domains/dom_fit.sh GPU;  pipeline/4_residual/domains/dom_sub.sh GPU
# 8    evaluation
python pipeline/8_compare/compare.py DS;  python tools/deck/build_deck.py data OUT.pptx
```

## Method

**Notation.**
- Particle $i$ has image $y_i$, fixed pose $R_i$, CTF $c_i$ and scale $a_i$; $P_i$ is the projection at $R_i$.
- A particle set $g$ gives half maps $A_g$ and $B_g$ (random halves). Each is low-passed to 10 Å and restricted to the envelope (64³).
- A "reproducible" quantity is a cross-product of independent halves, $\langle A-\bar A,\,B-\bar B\rangle$, so noise averages out.

### 1 · Envelope

The fitted 6MSM model (global fit, then a local TMD refit) defines a protein-only envelope, dilated 2 voxels.
- 5UAK can be superposed on the fitted TMDs to form a union envelope (`diagnostics/fit_union.py`, `union_env.py`).
- Domain regions from the same model: TMD1 1–380, NBD1 381–645, TMD2 840–1172, NBD2 1173–1480.

The consensus envelope is not used, because much of its extra density is micelle.

### 2 · Residual model (`4_residual/residual_ppca.py`)

The images are split into equal-count cells $c(i)$ along the current coordinate, and $m_c$ is the back-projected map of cell $c$. A $K$-direction PPCA is fitted to the whitened residuals in the 50–7 Å band:

$$y_i = a_i c_i P_i\Big(m_{c(i)}+\sum_{k=1}^{K} z_{ik}u_k\Big)+\varepsilon,\qquad
z_i\mid r_i\sim\mathcal N\big((W_iW_i^\top+I)^{-1}W_i r_i,\ (W_iW_i^\top+I)^{-1}\big),\quad W_i=[\tau_k a_i c_i P_i q_k]$$

- **EM.** The M-step is one $K\times K$ normal equation per Fourier voxel. The eigen-volumes are multiplied by the soft envelope (or a domain mask) and re-orthonormalised. Defaults: $K=8$, 30 iterations.
- **Two folds.** Directions learned on one half of the particles score the other half. Reproducibility is the principal-angle cosines between the folds' subspaces, and the folds are aligned by Procrustes.
- **Random cells** (`start.py rand`): every cell map is the consensus, so the fit is unconditioned. This is the first step of every network-free start.
- **`MCR_SUBSET=idx.npy`**: fits only those particles against their own mean map. This looks for variation *within* a window or class.

### 3 · Starting axis (`3_scores/start.py`, `MP_START`)

All network-free starts work on the unconditioned coordinates $z^{(0)}\in\mathbb R^{8}$ from step 2, standardised. Only column 0 of `pc.scores.npy` is used downstream. `MP_SEED` replicates the random parts.

| Start | Construction |
|---|---|
| `free` | linear PCA of $z^{(0)}$ (3DVA-like); PC1 is the axis |
| `vq` (codebook) | 1. k-means of $z^{(0)}$ into 300 cells.<br>2. Back-project each cell's half maps (10 Å, envelope) and run a weighted PCA over the 300 cell maps.<br>3. A particle's score is the mean map-PC projection of its 8 nearest cell centres.<br>The axis is defined by *map* variance, not coordinate variance. |
| `diffmap` | spectral embedding (k-NN graph, $k=30$) of a 30 k subsample, extended to all particles by distance-weighted 15-NN regression; a nonlinear manifold coordinate |
| `mlp` | denoising autoencoder 8→128→128→**1**→128→128→8 (SiLU, input noise 0.1, 40 epochs); the bottleneck is a principal curve through $z^{(0)}$ |
| `latent` (optional) | PCA of a trained cryoDRGN encoder latent (free part only for the conditional RAE) |
| unset (optional) | cryoDRGN volume-PC route: decode 2000 latent medoids, weighted PCA of the decoded maps, particle score = mean over $k$ nearest medoids |

### 4 · Sharpening (`4_residual/sharpen.py`)

The start is refined against the images:
1. Fit the residual model conditioned on the start. Take 24 bins along it, with end maps $e_{lo}$ and $e_{hi}$ and $u=e_{hi}-e_{lo}$.
2. Place 200 cells in (start, $z$) and project each cell map onto the end-to-end line: $t_c=\langle X_c-e_{lo},u\rangle/\lVert u\rVert^2$.
3. Fit $t_c\approx\beta_0+\beta^\top \bar f_c$ cross-fitted between halves A and B. The sharpened coordinate is $s_i=\beta^\top f_i$.

### 5 · Fully conditioned search (`5_search/search.py`)

Rounds $r=1,\dots,R_{max}$, with coordinates $S$ and weights $w$ ($w_1=1$).

1. Make 64 equal-count cells conditioned on all of $S$, and refit the residual model to get $z\mid S$.
2. Back-project 400 k-means cells in $(S\,\mathrm{diag}(w),\ 0.5z)$. Each cell subtracts the $n$-weighted mean of its 8 nearest cells in $S$, giving $\tilde X_c$ and $\tilde z_c$.
3. The most reproducible direction solves

$$\Big(\sum_c n_c\tilde z_c\tilde A_c^\top\Big)\Big(\sum_c n_c\tilde z_c\tilde B_c^\top\Big)^{\!\top} w=\lambda\,\Big(\sum_c n_c\tilde z_c\tilde z_c^\top\Big)\,w .$$

   The null is 200 permutations of $\tilde z$ among cells with similar $S$. Stop at $p>0.05$.
4. Split the map effect of $x=w^\top z$ into a tangent part and a remainder.
   - The tangent part sharpens the existing coordinates.
   - A reproducible remainder ($p_{off}\le0.05$) makes $x$ a new coordinate with weight $\sqrt{\lambda/T_{main}}$.

### 6 · Classes and K (`6_classes`)

300 cells in $S\,\mathrm{diag}(w)$ are grouped by $n$-weighted k-means **on the cell maps**, for K = 1..10. A K passes when:
- every class reproduces between halves ($sh_k \ge 0.9$);
- every class's own off-blend component reproduces ($own\_sh_k \ge 0.5$);
- the smallest class is ≥ 4 % of the particles.

The refined set is every passing K, plus the elbow, plus one boundary K.

**Soft windows (`6_classes/windows`).** These are an alternative to hard classes for a state that sits between others along an axis. The axis can be any pipeline coordinate $S_j$ (or a domain coordinate). Particle $i$ is kept with probability $\exp(-(r_i-c)^2/2\sigma^2)$, where $r_i$ is its rank.
1. **Position:** $\sigma=0.10$ at $c=0.1,\dots,0.9$. Pick the interior window farthest from the line between the two end maps.
2. **Size:** at that $c$, $\sigma=0.04$–$0.22$. Pick the size whose map changes least relative to its neighbouring sizes (stability), so the size is not fixed in advance.

Both picks are label-free.

**Per-domain fits (`4_residual/domains`).** The residual model with the eigen-volumes confined to one domain (dilated about 8 Å, inside the envelope), run either on the whole dataset or inside a window (`MCR_SUBSET`). A reproducible domain component can be used as a window axis or split along (`sub_split.py`).

### 7 · Refinement (`7_refine`)

Each class or window is exported as a STAR file on the 128 px stack and imported **without poses**, then refined by ab initio (1 class) → NU with fixed parameters (P17 J451 / J452). The heterorefinement classes go through the identical route. `remote/dispatch.py` spreads the chains over several cryoSPARC instances and fetches the maps back.

### 8 · Comparison (`8_compare`, evaluation only)

Every map is aligned to a common frame, hand-corrected and TMD-refitted onto the consensus $c_0$. At 10 Å inside the envelope, after a linear density-scale fit:

$$cc_k=\mathrm{corr}(x-c_0,\ C_k-c_0),\qquad d^2_k=\lVert x-C_k\rVert^2/\lVert C_a-C_b\rVert^2$$

$(a,b)$ is the most different reference pair. Sets are Hungarian-matched by $cc$, and the deck reports the nearest class by $d^2$.

$cc$ is ill-conditioned for a class close to the consensus (J4001 P7: $d^2=0.093$ from the consensus), so $d^2$ is the metric for such classes.

## Results so far (2026-11-26, preliminary)

**Starts.** All five starts were run at K = 2–5 (J399 K = 2–10) and refined.
- J4001: every start gives the same class maps (Hungarian cc 0.95–0.98). The `free`, `vq` and `mlp` seed replicates reproduce at K = 2.
- J2708 and J399: the network-free starts agree with each other (0.87–0.93). The cryoDRGN-latent start differs.

**Residual model.**
- The first-fit fold cosines are 0.90–0.91 on J4001, J2708 and J399.
- J207 reaches only 0.49, so there is no reproducible signal to classify.
- In the search, only rounds 1–2 reproduce (round 3 too on J399). Later coordinates are mostly noise, with correlations up to 0.68.

**Envelope.** The wide-V classes (J2708 C09, J399 C11) are not recovered by the protein-envelope runs.
- Their extra density is 68–69 % outside even the 5UAK + 6MSM union.
- 80–93 % of that density is in the membrane belt, mostly more than 20 Å from protein. It looks like micelle, so the protein-only envelope is kept.

**P7 (J4001).**
- No coordinate separates P7 particles (AUC 0.50–0.57).
- By $d^2$, two maps are closest to P7: the middle K = 3 class (0.073) and the S0 middle window (0.077, the label-free stage-1 pick). For comparison, the consensus is at 0.093.
- The S1, S2 and NBD1-axis windows do not beat the consensus (0.10–0.11).

**Per-domain fits (J4001).**
- On the whole dataset, TMD1, NBD1 and TMD2 each have one reproducible component (cosine ≈ 0.9).
- NBD1's component is the least redundant with the whole-molecule coordinates (canonical corr 0.69 / 0.28). It raises P7 AUC from 0.57 to 0.64.
- Inside a single window (about 48 k particles), no domain component reproduces (≤ 0.49).

**NBD1 density** follows the main axis: 0.5× consensus at the P6 end and 1.6× at the P8 end (corr 0.99). The P8-like maps match the P8 reference in NBD1 (cc 0.95 at 6 Å).

**Earlier cryoDRGN-RAE route (11-24).**
- J4001 K = 3 matched P8 0.98 and P6 0.99, with a P7-leaning class at 0.53–0.60.
- J4624 K = 3 matched C09 0.82 and C08 0.84; K = 4 added C07 (0.75).
- Model variants changed particle labels (ARI ≈ 0.35) but not class maps (cc ≥ 0.99).

## Open questions

| Question | Current state |
|---|---|
| Which start to use by default | The network-free starts agree on J4001 / J2708 / J399; `free` is the simplest and `vq` measures map variance directly |
| How many search coordinates to keep | Only rounds 1–2 reproduce; a fold-cosine or seed rule for keeping rounds is to be fixed |
| Windows at every new axis | Implemented for S0–S2 and a domain axis; only S0 gives a P7-closer map so far |
| Domain variation inside a state | Not detectable at about 48 k particles per window; whole-dataset domain axes are the usable form |
| J207 | No reproducible residual signal; needs more particles or a different band |
| Class-count thresholds | split-half 0.9 is strict for small datasets (J207, 61 k) |
