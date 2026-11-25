# cryoem-classification

Label-free conformational classification of hCFTR cryo-EM particles. A cryoDRGN model (RAE by default) is trained on consensus NU poses.
Its latent space is used only to order particles. Every later decision is made on **back-projected half maps of the real images with fixed poses**:
a main coordinate, residual directions found conditionally on everything already explained, classes in that multi-coordinate space, and
a class count chosen by split-half and seed reproducibility. Classes are then refined in cryoSPARC (ab initio + NU). Heterorefinement classes
are used **only for evaluation**: they are refined by the same route and compared at the end.

## Layout

```
config/      env.sh (site paths), datasets.tsv (stack, poses, CTF, envelope, mask, voxel size, max rounds), heteroref.json, examples
lib/         fastbp.py (GPU back-projection / half maps / simulation), maps.py (envelope crop, class-map algebra), cryo_utils.py, mapdiff.py
pipeline/    0_consensus  1_envelope  2_train  3_scores  4_residual  5_search  6_classes  7_refine  8_compare
             run_model.sh (steps 3-6 for one model), pipe_worker.sh (queue), auto_select.sh (K scan -> refinement, hands-off)
reference/   PDB models, per-dataset envelopes, cryoSPARC job ids, K-scan / reproducibility results (see reference/README.md)
legacy/      earlier approaches (latent trees, diagnostics); not used
```

Datasets: `J4001_gP25_WT`, `J4624_eP37_WT_T2a`, `J207_hP12_IDOR4`, `J2708_eP30_F508del_E1371W`, `J399_hP7_E1371Q_ATP`.
Large data stays on disk (`$RES`, `$WORK` in `config/env.sh`). Run with the `cryodrgn` conda env (cryoDRGN 4.3). The `cx_*` / `fit_pdb.py` scripts run in headless ChimeraX.

## Running

```bash
source config/env.sh
# 0  consensus poses + cryoDRGN inputs (128 px stack, ctf.pkl, poses.pkl)
python pipeline/0_consensus/prepare.py NAME && python pipeline/0_consensus/write_star.py NAME   # + cs_pose_jobs.py in cryosparcm cli
pipeline/0_consensus/consensus.sh NAME GPU
# 1  envelopes (consensus + fitted PDB);  add a row to config/datasets.tsv
pipeline/1_envelope/envelope.sh GPU DS STACK POSES CTF
# 2  training: one worker per GPU on any machine, jobs from queue.txt (config/train_queue.example)
pipeline/2_train/train_worker.sh machine.conf GPU
# 3-6  per trained model (or: lines "DS NAME MODELDIR 50 [OPTS]" in $WORK/pipe_queue.txt + pipe_worker.sh GPU)
pipeline/run_model.sh GPU DS rae_base $WORK/models/DS/rae_base 50
# 6-7  K scan, seed reproducibility, refinement of the candidate K set (+ heterorefinement classes for the comparison)
pipeline/auto_select.sh "DS/rae_base:auto DS/rae_s7: DS/rae_s101:";  pipeline/7_refine/submit.sh DS hetero
# 8  comparison with the heterorefinement maps (evaluation only)
python pipeline/8_compare/compare.py DS
```

## Method

**Notation.** Particle $i$ has image $y_i$, fixed pose $R_i$, CTF $c_i$ and scale $a_i$. $P_i$ is the projection at $R_i$.
For a particle set $g$, back-projection gives half maps $A_g$ and $B_g$ (random halves). Each map is low-passed to 10 Å and restricted to the envelope (64³ grid).
All "reproducible" quantities are cross-products of independent halves, $\langle A-\bar A,\,B-\bar B\rangle$, so noise averages out.

**2 · Training (RAE, Ghosh et al. 2020).** Deterministic encoder $z_i = E(y_i)$ with decoder $D$, trained on fixed NU poses:

$$\mathcal L=\sum_i \lVert y_i - c_i P_i D(z_i)\rVert^2 + \frac{1}{N_\text{pix}}\Big(\tfrac{\beta}{2}\lVert z_i\rVert^2+\lambda\,\lVert \partial D/\partial z\rVert_F^2\Big)$$

Default settings: $z\in\mathbb R^{10}$; encoder and decoder 1024×3; $\beta=0.1$; $\lambda=0.00243$; 128 px; 50 epochs; batch 64; seed 30112.
Replica models use the same settings with other seeds (7, 101, 202, 303). VAE ($\beta_{KL}=0.1$) and plain-AE variants use the same trainer.

**3 · Volume-PC scores.** Decode \(M = 2000\) k-means medoids of \(z\) at \(64^3\) resolution and smooth each map (8 Å FWHM).

Weight medoid \(m\) by

$$
w_m \propto 1 + \left|\{i : \mathrm{nearest}(z_i)=m\}\right|,
$$

and run a weighted PCA on the medoid maps, giving projections \(P_{mj}\).

A particle's score is the mean projection of its \(k\) nearest medoids in latent space (\(k\) chosen by leave-one-out cross-validation)

**4 · Residual model + sharpening.** The images are split into equal-count cells $c(i)$ along the current coordinate. A $K$-direction probabilistic PCA is then fitted to the whitened image residuals:

$$y_i = a_i c_i P_i\Big(m_{c(i)}+\sum_{k=1}^{K} z_{ik}u_k\Big)+\varepsilon,\qquad
z_i\mid r_i\sim\mathcal N\big((W_iW_i^\top+I)^{-1}W_i r_i,\ (W_iW_i^\top+I)^{-1}\big),\quad W_i=[\tau_k a_i c_i P_i q_k]$$

- **M-step:** one $K\times K$ normal equation per Fourier voxel. The eigen-volumes are kept inside the envelope.
- **Defaults:** $K=8$, band 50–7 Å, 30 EM iterations.
- **Two folds:** directions learned on one half of the particles score the other half. The fold subspaces are compared (principal-angle cosines) and aligned by Procrustes.

The first coordinate is then **sharpened**:
1. Take 24 equal-count bins along PC1. Let $e_{lo}$ and $e_{hi}$ be the mean maps of the two end bins on each side, and $u = e_{hi}-e_{lo}$.
2. Place 200 cells in $(\text{PC1}, z)$ and project each cell map onto the end-to-end line: $t_c=\langle X_c-e_{lo},u\rangle/\lVert u\rVert^2$.
3. Fit $t_c\approx\beta_0+\beta^\top \bar f_c$ by weighted least squares, on half A scored on half B and vice versa.
4. The sharpened coordinate is $s_i=\beta^\top f_i$.

**5 · Fully conditioned search.** Rounds $r=1,\dots,R_{max}$. Coordinates $S$ (columns) have weights $w$, with $w_1=1$.

1. Make 64 equal-count cells conditioned on all of $S$ (recursive bisection along the leading direction of $S\,\mathrm{diag}(w)$). Refit the residual model, giving $z$ conditional on $S$.
2. Back-project 400 k-means cells in $(S\,\mathrm{diag}(w),\ 0.5z)$ as half maps $A_c$ and $B_c$.
3. Remove what $S$ already predicts. Each cell subtracts the $n$-weighted mean of its 8 nearest cells in $S$: $\tilde X_c = X_c-\sum_{c'\in N_8(c)}\omega_{cc'}X_{c'}$, and likewise $\tilde z_c$. The most reproducible direction in $z$ then solves

$$\Big(\sum_c n_c\tilde z_c\tilde A_c^\top\Big)\Big(\sum_c n_c\tilde z_c\tilde B_c^\top\Big)^{\!\top} w=\lambda\,\Big(\sum_c n_c\tilde z_c\tilde z_c^\top\Big)\,w .$$

   The null distribution comes from 200 permutations of $\tilde z$ among cells with similar $S$. Stop if $p=P(\lambda_{null}\ge\lambda)>0.05$.
4. Split the map effect $v$ of $x=w^\top z$ into its projection on the tangents $T$ of the existing coordinates and a remainder $v_\perp$.
   - The tangent part **sharpens** them: $S\leftarrow S+x\,\beta^\top$.
   - If the remainder is reproducible beyond its null ($\langle v_{A\perp},v_{B\perp}\rangle$, $p_{off}\le 0.05$), $x$ becomes a **new coordinate** with weight $w_{new}=\sqrt{\lambda/T_{main}}$. Here $T_{main}$ is the reproducible between-cell variance of round 1.
   - No orthogonality is imposed.

The rounds kept are decided by seed replicas: keep round $r$ while the median best $\lvert cc\rvert$ of its mode map with the seed models' mode maps is ≥ 0.75.

**6 · Classes and K.** Back-project 300 cells in $S\,\mathrm{diag}(w)$. Group the cells with n-weighted k-means **on the cell maps**, $\min\sum_c n_c\lVert X_c-\mu_{k(c)}\rVert^2$ (20 restarts), for K = 1..10. The class maps are $M_k=\sum_{c\in k}n_cX_c/\sum_{c\in k}n_c$.

| Quantity | Definition |
|---|---|
| within-class reproducible variance | $W_K=\sum_k\sum_{c\in k}n_c\langle A_c-\bar A_k,B_c-\bar B_k\rangle$ → elbow, gain $(W_{K-1}-W_K)/W_{K-1}$ |
| own (off-blend) component | $o_k = M_k-\Pi_{\mathrm{aff}\{M_j\}_{j\ne k}}M_k$: what class $k$ has that is not a blend of the others |
| split-half | $sh_k=\mathrm{cc}(M^A_k-\bar M^A,\ M^B_k-\bar M^B)$; $own\_sh_k=\mathrm{cc}(o^A_k,o^B_k)$ |
| seed reproducible / distinct | Hungarian-matched class-map cc ≥ 0.8 in most seeds / own-component cross-seed cc ≥ 0.5 |

A value of K **passes** when $\min_k sh_k\ge0.9$, $\min_k own\_sh_k\ge0.5$ and the smallest class is ≥ 4 % of the particles. When seed models exist, every class must also be distinct across seeds.
Refined candidates = every passing K, plus the elbow, plus (largest passing K + 1) as a boundary check (`kselect.py`).

**7 · Refinement.** One STAR per class is rewritten to the 128 px stack and imported **without poses**. Each is refined by ab initio (1 class) → NU, with parameters from P17 J451 / J452.
The heterorefinement classes go through the identical route.

**8 · Comparison (evaluation only).**
1. Bring every map into the deck frame and fix the hand, then refit on the TMD core onto the consensus $c_0$ (n-weighted mean of the refined heterorefinement maps).
2. At 10 Å inside the envelope, after a linear density-scale fit:

$$cc_k=\mathrm{corr}(x-c_0,\ C_k-c_0),\qquad d^2_k=\lVert x-C_k\rVert^2/\lVert C_a-C_b\rVert^2$$

   $(a,b)$ is the most different pair of heterorefinement classes. Each refined set is Hungarian-matched to the heterorefinement classes by $cc$.

## Results so far (2026-11-24)

- **Seed-reproducible search rounds:** J4001 4, J4624 5.
- **Reproducible and distinct classes:** up to K = 5 on both. K = 4 adds an intermediate (P7- / C07-rich) class, and K = 5 splits P6 / C08.
- **J4001 K = 3 vs heterorefinement:** P8 0.98, P6 0.99, P7-leaning class 0.53–0.60. Alt-NBD1 merges with P6.
- **J4624 K = 3:** C09 0.82, C08 0.84, C06 + C10 merged 0.92. K = 4 adds C07 at 0.75.
- **Label-free K scan:**
  - J207: elbow 3; only K = 2 passes; K = 3 own_sh is 0.85. The split-half 0.9 cutoff is strict for 61 k particles.
  - J2708: elbow 3; K = 2–5 pass.
  - J399: pending.

  The candidate sets and the heterorefinement classes of all three datasets are being refined.
- **Robustness:**
  - Model variants (seed, z-dim, β, λ, VAE, AE) change particle labels (ARI ≈ 0.35) but not class maps (cc 0.99–1.00 on 2 of 3 classes).
  - Low-pass 15 Å: maps cc ≥ 0.95.
  - Keeping only the reproducible rounds (`MP_KEEP`): maps cc 0.95–1.00.

## Open decisions and ongoing tests

| Decision | Current default | Tested / running | Status |
|---|---|---|---|
| Model family | RAE | VAE (β 0.1), plain AE, z 4 / 16, β 0.01, λ 0, width 256 | maps agree (cc ≥ 0.99 on 2 of 3 classes); labels do not |
| Training resolution | 128 px | 256 px stack (J4001 running) | open |
| Number of seed replicas | base + 2–4 | J207 / J2708 / J399 replicas training | needed for round and K reproducibility |
| Envelope | fitted PDB, dilated 2 voxels | consensus envelope (`*c` rows), dilation 0 / 4 (`*d0`, `*d4`) | stress test in progress |
| Map resolution of steps 4-6 | 10 Å, 64³ | 15 Å (cc ≥ 0.95); 6 / 4 Å at 128³ (ARI 0.5–0.65, off-blend rule gives K = 10, likely noise) | class-map cc at 6 / 4 Å still to compute |
| Residual model | K = 8, 50–7 Å | `MP_K`, `MP_HI`, `MP_LO` | default kept |
| Rounds kept | all rounds with p ≤ 0.05 | seed rule: keep while \|r\| ≥ 0.75; `MP_KEEP` test (maps cc 0.95–1.00) | little effect on maps |
| Class-count rule | pass rule above + candidates | split-half 0.9 cutoff vs dataset size (J207) | user decision: thresholds, and whether seed distinctness is required |
| Particle reassignment | off | `MP_REASSIGN`: real 6–8 % change per round vs null 13–15 %; margin 5–6 % vs 1.6 %; ARI vs start 0.22 | real signal but uncertain labels; maps robust; optional |
| Refinement route | import without poses → ab initio → NU, 128 px | (possible) local refinement from consensus poses; full-resolution stack | open |
