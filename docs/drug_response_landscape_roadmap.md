# Drug-conditioned conformational landscape of CFTR — research roadmap

Status: planning + phase-1 tooling in place (single condition).
Last updated: 2026-07-15.

## One-line thesis

Model each drug's effect as a **perturbation of a continuous conformational
free-energy surface** inferred from cryo-EM, rather than as a change in discrete
class occupancies. Deliverable: a **drug perturbation field** ΔF(z) over a shared
cryoDRGN latent, validated against known biology and (later) MD.

This reframes the question from *"what structure does drug X produce?"* to
*"how does drug X reshape the whole conformational ensemble?"*.

## Why this is novel (and defensible)

- cryoDRGN/3DVA + MD integration: discussed in the field, not the novel part.
- Docking/MD/cryo-EM for CFTR modulators: established.
- **Learning a drug-conditioned continuous free-energy landscape from cryo-EM
  ensembles across multiple drug conditions: no published example found.**
- Framing drug effects as ΔF over a continuous surface (vs discrete class
  occupancy shifts): largely absent from current CFTR work.

Our own evidence already supports the continuous framing: at converged D=256 the
CFTR latent is a single continuous basin with reproducible populations; the
Wang/Hunt GCER paper itself calls the well-ordered states "a continuum of
dynamically interconverting conformational states."

## The single falsifiable headline claim (pre-registered)

The Wang/Hunt 2022 GCER paper already gives the ground truth to reproduce:
**VX770 alone does not measurably reshape the ensemble; VX445 drives a large
redistribution toward the narrow (N3 / active-like) region.**

So we commit in advance:
> Our continuous ΔF pipeline must recover ΔF(VX770) ≈ bootstrap null floor and
> ΔF(VX445) large and localised to the narrow region.

Recovering a known biological result with a new *continuous* method is the
strongest validation and disarms the "just a framework" critique.

## The three connected ideas

### Idea 1 — EM ↔ MD cross-validation (safest)
Compare **reaction coordinate + free-energy profile**, not discrete labels:
does MD independently reproduce the coordinate and F(coordinate) that cryoDRGN
infers (NBD rocking SC→AC→AO, gate opening, TM8/TM11 ordering, exit-portal,
NBD1 association)?

### Idea 2 — Drug-conditioned landscape: ΔF(z)
One shared latent; estimate p(z | condition) for each drug; report
ΔF(z), Δoccupancy(z), Δentropy(z) — a continuous **drug perturbation field**.

### Idea 3 — Distribution transport (the honest "transition operator")
cryo-EM observes p(z | condition), never (z_t, z_{t+1}). So learn a **distribution
shift** T: p(z|vehicle) → p(z|drug) via entropic optimal transport (Sinkhorn),
not trajectories. ΔF(z) is the scalar potential; the OT displacement field is the
vector field. (Skip Schrödinger bridges — they imply dynamics we don't have.)

## The linchpin: a physical bridge coordinate

cryoDRGN `z` is an *image* embedding, not a physical CV; MD atomic frames cannot
be placed in it directly. Everything above is undefined until the latent is tied
to a physical coordinate both cryo-EM volumes and MD frames can report.

Bridge = decode volumes along latent PCs → measure physical structural
descriptors on each → build a `z ↔ CV` calibration and anchor the paper's
states (SC/AC/AO, V17/V21/V23/V31/N3) onto the CV axes. Then F(z) is reported as
F(physical CV), directly comparable to MD's F(CV).

Model-free descriptors used now (no atomic model required):
- `mol_vol` — molecular volume above a contour (order / NBD1 density proxy).
- `Rg` — radius of gyration (openness).
- `anisotropy` — inertial elongation (V-shape opening).
- `sym_max` — reflection symmetry about COM (Symmetric-Closed vs Asymmetric).
- `lobe_asym` — density asymmetry between halves (TMD/NBD partition proxy).

The exact **inter-TMD rotation angle** (the paper's naming axis) needs a docked
atomic model; a hook (`--atomic-model`) is reserved for the deposited G551D
coordinates. Model-free descriptors proxy the same physics until then.

Implemented in: `scripts/cryodrgn/cryodrgn_bridge_coordinate.py`.

## Executable now (single condition), so the drug work is plug-and-play

1. **Bridge-coordinate calibration** — DONE (tooling). Run on J1442/J264 to
   re-express the latent in physical descriptors and anchor SC/AC/AO.
2. **Dry-run the drug-response pipeline on pseudo-conditions** — split one dataset
   into artificial conditions (random halves = the null floor; spiked
   subpopulation = a known signal) and run p(z|c) → ΔF(z) with bootstrap null →
   Sinkhorn OT displacement. Validates the harness end-to-end. [TODO: script]
3. **MD-on-manifold projection spec** — define which physical CVs are measured on
   decoded volumes and on MD frames, so a trajectory can be tested immediately
   once MD exists. [TODO: spec + measurement code shared with #1]
4. **Pre-register** the VX770-null / VX445-shift target above.

## Blocked until data / assets arrive (see email to Will)

- **Multi-condition particle stacks** (raw .mrcs + consensus poses + CTF) for the
  drug conditions, to train ONE joint cryoDRGN on pooled particles.
- **Refined atomic models** (deposited G551D coordinates) for exact inter-TMD
  angle / domain-resolved CVs and MD starting structures.
- **Half-maps** for true local-resolution / FSC validation.
- Confirmation of what the CryoSPARC "input groups" (e.g. J2694 groups 0–3)
  correspond to (sessions vs samples/conditions).

## Dataset selection (from GCER Population Tables.xlsx, Current Processing)

Group by **construct** (a shared latent is only meaningful within one construct —
differences must be drug, not construct). Colour = done (blue) / basically done
(white) / in-progress / low-priority. Pixel size is NOT a blocker: pool by
resampling every condition to ONE common downsampled box + Apix in cryoDRGN
`downsample` (Fourier crop handles 0.83 vs 0.84 trivially; 1.06 or super-res
0.42/0.53 just downsample to the common target).

### Construct A — E1371Q-6SS / D1247 (catalytically-dead, ATP always bound)
This is the user's existing cryoDRGN construct (J264 is E1371Q). Best for novelty + reuse.
- **ATP baseline (DONE):** hP7W1/yP55W2 (23apr14c ctrl-3hr, pix **0.84**) — matched-pixel
  best; also hP8 (EMPIAR 13665, pix 1.061, older).
- **ATP/IDOR4 (DONE):** hP12W1 (26may26f, pix **0.835**) — a drug-bound, finished dataset.
- **ATP "VX_K" (DONE):** eP3 (pix 1.047).
- **ATP/VX770 and ATP/VX770/VX445 (NOT collected/located = "???"):** ask Will — these
  complete a clean nested VX series on this construct.
- Nucleotide/time variants (MANTATP, +T8, +T2a, ATP-RT, +lpp, Sionna combos): later.

### Construct B — G551D / D1177 (the PUBLISHED Wang/Hunt paper series = validation anchor)
Known drug-response answer -> reproduce VX770-null / VX445-shift.
- **ATP baseline (DONE):** hP9/yP14W6, **EMPIAR 13267**, pix 0.8385
  (volga:/mnt/data1/CFTR/19dec16a1/rawdata) — the "control from the paper" being reprocessed.
- **ATP/VX770:** yP14W4 (in-progress, pix 0.83).
- **ATP/VX770/VX445:** gP10W2 (bioRxiv/paper, pix 0.83) [+ eP25 super-res 0.423].
- **peATP/VX770/VX445/VX809:** eP25W12 (pix 0.8255) — confounded 4th arm.

### Construct C — F508del-E1371Q-6SS (most disease-relevant, hardest/confounded)
- ATP/VX809: eP30 (DONE, EMPIAR 13234). MANTATP/VX809/VX770/VX445: eP25W9 (in-progress).
  Mixed nucleotide+correctors = confounded; save for last.

### Priority order for training
1. **hP9 G551D/ATP (EMPIAR 13267)** — already reprocessing; validation baseline. Finish
   cryoDRGN + run latent_gmm uncertainty (John's ask).
2. **E1371Q ATP vs IDOR4: hP7W1 (0.84) + hP12W1 (0.835)** — both DONE, matched pixel, same
   construct as J264 -> FIRST real drug-response landscape (joint cryoDRGN -> ΔF + OT).
3. **G551D VX series: yP14W4 (VX770) + gP10W2 (VX770/VX445)** -> completes the published
   validation series (pre-registered VX770-null / VX445-shift).
4. Request the "???" E1371Q/VX770 + VX770/VX445 collections from Will; F508del triple later.

NOTE: confirm whether the existing J264 cryoDRGN run == one of the E1371Q/ATP controls
(hP8 18nov23a or hP7W1 23apr14c). If so the E1371Q ATP baseline is already trained.

## Method guardrails (do not overclaim)

- Shared latent REQUIRES joint training on POOLED particles with **no condition
  input** (post-hoc p(z|condition)); a conditional decoder can "explain away" the
  drug effect. Cross-run canonical-correlation reproducibility (~0.98) only holds
  for the SAME particles — it does not license comparing independently-trained
  latents across different particles.
- **Imaging-confound trap:** different conditions = different grids/sessions, so
  verify the manifold separates by conformation, not defocus/CTF/batch
  (extend `cryodrgn_overfit_check.py` to test PC vs condition vs imaging).
- **ΔF needs a null floor:** half-vs-half ΔF within one condition = the noise
  floor given finite sampling + KDE bandwidth; real features must exceed it.
- The four Hunt conditions are **not a clean factorial** (only ATP → +VX770 →
  +VX770+VX445 holds nucleotide constant; the pATP/VX809 arm is confounded) and
  use **VX770/VX445/VX809 (+pATP), not VX661**. They are the **G551D** construct —
  do not pool across constructs (J264 etc.) into one manifold.
- Claim population/occupancy/ΔF shifts (cryo-EM supports this); treat
  pathways/kinetics as MD-validated only.
