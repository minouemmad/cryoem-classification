"""Apply the GMM pipeline to the ENSEMBLE classifier and export per-class
particle stacks ready for CryoSPARC reconstruction.

This ties together the three earlier analyses:

1. ``gmm_referee_compare.py`` showed the honest J1442 posteriors define the
   conformational manifold; J1069's overconfident labels cut across it.
2. ``ensemble_reweight.py`` combined the two experts and showed an overconfident
   expert hijacks the ensemble's hard decisions unless it is temperature-
   recalibrated first.

Here we build the **calibrated equal-weight ensemble posterior** (the principled
combination of both jobs), fit the GMM to it (the same GMM machinery used
elsewhere), and turn the GMM's per-particle responsibilities into three
disjoint particle stacks (P6/P7/P8) that you can import into CryoSPARC and run
"Homogeneous Reconstruction Only" on.

Why recompute the ensemble here (instead of reading the .npz)?
    CryoSPARC uids exceed 2**53, so a float round-trip loses them. We rebuild the
    ensemble straight from the two source stacks, keeping uids as uint64/ints, so
    the exported stacks match the J1442 particle blob + passthrough exactly.

Pipeline
--------
    J1442 honest posteriors  ─┐
                              ├─ calibrate J1069 to J1442 entropy ─► linear pool (λ=0.5)
    J1069 overconfident  ─────┘                                         │
                                                                        ▼
                                              GMM in ALR(ensemble) space (k=3)
                                                                        │
                                       per-particle responsibilities r_nk
                                                                        │
                       hard subset {n : argmax_k r_nk == k}, scale := r_nk  (β-sharpenable)
                                                                        │
                              per-class  *.cs + *_passthrough.cs  ──►  CryoSPARC

Outputs (--outdir/exports_ensemble_gmm)
---------------------------------------
* ``ensembleGMM_{P6,P7,P8}.cs`` + ``..._passthrough.cs``  — import these.
* ``ensemble_gmm_populations.csv``      hard/soft/deconvolved populations.
* ``ensemble_gmm_assignment_uid.csv``   uid, ensemble class, GMM class, max-resp.
* ``ensemble_gmm_responsibilities.npy`` (N, K) GMM responsibilities.
* ``SUMMARY.md``
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, linear_sum_assignment
from sklearn.metrics.cluster import contingency_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gmm_pipeline import (  # noqa: E402
    alr_transform,
    deconvolve_populations,
    fit_gmm,
    gmm_diagnostics,
    load_posteriors,
    observed_populations,
    soft_posterior_confusion,
)

_EPS = 1e-12


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--j1442-cs", default="data/cryosparc_P25_J1442_00000_particles.cs",
                   help="Honest expert (also the particle stack + reference space)")
    p.add_argument("--j1069-cs", default="data/cryosparc_P25_J1069_00042_particles.cs",
                   help="Overconfident expert (recalibrated before pooling)")
    p.add_argument("--passthrough-cs",
                   default="data/cryosparc_P25_J1442_passthrough_particles_all_classes.cs",
                   help="J1442 passthrough (poses + per-particle scale field)")
    p.add_argument("--n-dummies", type=int, default=6)
    p.add_argument("--protein-idx", type=int, nargs="*", default=None)
    p.add_argument("--lam", type=float, default=0.5,
                   help="Ensemble mixing weight (1=J1442 honest, 0=J1069). Default 0.5.")
    p.add_argument("--no-calibrate", action="store_true",
                   help="Skip temperature recalibration of J1069 before pooling")
    p.add_argument("--beta", type=float, default=1.0,
                   help="Sharpen the exported scale weight: w := r^beta normalised "
                        "(1 = raw GMM responsibility).")
    p.add_argument("--combine", choices=["replace", "multiply", "none"], default="replace",
                   help="replace = scale := weight; multiply = orig*weight; none = leave scale")
    p.add_argument("--scale-field", default="alignments3D/alpha")
    p.add_argument("--name-prefix", default="ensembleGMM")
    p.add_argument("--covariance", default="full")
    p.add_argument("--outdir", default="results_J1069")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _norm_rows(p):
    p = np.asarray(p, float)
    return p / p.sum(axis=1, keepdims=True).clip(min=_EPS)


def _mean_entropy(p, k):
    p = np.clip(p, _EPS, 1.0)
    return float((-(p * np.log(p)).sum(axis=1) / np.log(k)).mean())


def _temperature_to_match_entropy(p, target, k):
    logp = np.log(np.clip(p, _EPS, 1.0))
    apply = lambda T: _norm_rows(np.exp(logp / T))
    f = lambda T: _mean_entropy(apply(T), k) - target
    lo, hi = 1e-3, 1e3
    if f(lo) * f(hi) > 0:
        T = lo if abs(f(lo)) < abs(f(hi)) else hi
        return T, apply(T)
    T = brentq(f, lo, hi, xtol=1e-4)
    return T, apply(T)


def _align_labels(reference, candidate, n):
    C = contingency_matrix(reference, candidate)
    r, c = C.shape
    m = max(r, c)
    Cs = np.zeros((m, m)); Cs[:r, :c] = C
    row, col = linear_sum_assignment(-Cs)
    mapping = {int(cand): int(ref) for ref, cand in zip(row, col)}
    return np.array([mapping.get(int(x), int(x)) for x in candidate]), mapping


def main():
    args = parse_args()
    out = Path(args.outdir) / "exports_ensemble_gmm"
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1. build the calibrated ensemble posterior (exact uids) ----------
    print("[1/5] Loading both experts")
    pA = load_posteriors(args.j1442_cs, protein_idx=args.protein_idx,
                         n_dummies=args.n_dummies).protein_only()
    pB = load_posteriors(args.j1069_cs, protein_idx=args.protein_idx,
                         n_dummies=args.n_dummies).protein_only()
    k = pA.n_protein
    labels = [f"P{int(c)}" for c in range(6, 6 + k)] if k == 3 else [f"C{i}" for i in range(k)]

    bmap = {int(u): i for i, u in enumerate(pB.uid.tolist())}
    uids = [int(u) for u in pA.uid.tolist()]
    order = [bmap.get(u, -1) for u in uids]
    keep = np.array([o >= 0 for o in order])
    order = np.array([o for o in order if o >= 0])
    rA = pA.posterior[keep]
    rB = pB.posterior[order]
    uids = [u for u, kp in zip(uids, keep) if kp]
    N = len(rA)
    print(f"      shared protein particles: {N:,}")

    if args.no_calibrate:
        rB_use, T = rB, None
        print("      (calibration skipped)")
    else:
        targetH = _mean_entropy(rA, k)
        T, rB_use = _temperature_to_match_entropy(rB, targetH, k)
        print(f"      J1069 temperature-recalibrated: T*={T:.2f} "
              f"(entropy {_mean_entropy(rB,k):.3f} -> {_mean_entropy(rB_use,k):.3f})")

    r_ens = _norm_rows(args.lam * rA + (1.0 - args.lam) * rB_use)
    ens_hard = r_ens.argmax(axis=1)
    print(f"      ensemble (lam={args.lam}) mean max-post={r_ens.max(1).mean():.3f} "
          f"entropy={_mean_entropy(r_ens,k):.3f}")

    # ---- 2. apply the GMM to the ensemble posterior -----------------------
    print("[2/5] Fitting GMM to ALR(ensemble) posteriors")
    X = alr_transform(r_ens)
    res = fit_gmm(X, n_components=k, init_hard=ens_hard,
                  covariance_type=args.covariance, random_state=args.seed)
    diag = gmm_diagnostics(res)
    # align GMM components to ensemble class ids so P6/P7/P8 stay meaningful
    gmm_hard_aligned, mapping = _align_labels(ens_hard, res.hard_labels, k)
    # reorder responsibility columns to the aligned class ids
    inv = {v: kk for kk, v in mapping.items()}      # aligned_id -> gmm_col
    resp = res.responsibilities[:, [inv[i] for i in range(k)]]
    max_resp = resp.max(axis=1)
    print(f"      GMM converged={res.converged}  BIC={res.bic:.1f}  "
          f"mean max-resp={max_resp.mean():.3f}")

    # ---- 3. populations ---------------------------------------------------
    print("[3/5] Populations")
    pi_ens_hard = observed_populations(ens_hard, k)
    pi_gmm_hard = observed_populations(gmm_hard_aligned, k)
    pi_soft = r_ens.mean(axis=0)
    C_soft = soft_posterior_confusion(r_ens)
    pi_deconv = deconvolve_populations(pi_ens_hard, C_soft)
    pop_df = pd.DataFrame({
        "class": labels,
        "ensemble_hard": pi_ens_hard,
        "gmm_hard": pi_gmm_hard,
        "ensemble_soft_mean": pi_soft,
        "deconvolved": pi_deconv,
    })
    pop_df.to_csv(out / "ensemble_gmm_populations.csv", index=False)
    print(pop_df.round(4).to_string(index=False))

    # per-particle assignment table + responsibilities
    np.save(out / "ensemble_gmm_responsibilities.npy", resp)
    pd.DataFrame({
        "uid": np.asarray(uids, dtype=np.uint64),
        "ensemble_class": [labels[c] for c in ens_hard],
        "gmm_class": [labels[c] for c in gmm_hard_aligned],
        "gmm_max_resp": max_resp,
    }).to_csv(out / "ensemble_gmm_assignment_uid.csv", index=False)
    agree = float((ens_hard == gmm_hard_aligned).mean())
    print(f"      ensemble-argmax vs GMM-argmax agreement: {agree:.4f}")

    # ---- 4. export per-class stacks for CryoSPARC -------------------------
    print(f"[4/5] Exporting per-class stacks from {args.j1442_cs}")
    cs_orig = np.load(args.j1442_cs)
    pass_orig = np.load(args.passthrough_cs)
    if args.scale_field not in pass_orig.dtype.names:
        sys.exit(f"ERROR: scale field {args.scale_field} not in passthrough")
    cs_row = {int(u): i for i, u in enumerate(cs_orig["uid"])}
    pt_row = {int(u): i for i, u in enumerate(pass_orig["uid"])}
    scale_dtype = pass_orig.dtype[args.scale_field]

    # weight = (GMM responsibility for the assigned class)^beta, normalised per
    # particle so the assigned class carries the soft confidence.
    rb = np.power(np.clip(resp, _EPS, None), args.beta)
    w = rb / rb.sum(axis=1, keepdims=True)

    manifest = []
    for kk in range(k):
        sel = gmm_hard_aligned == kk
        uids_k = [u for u, s in zip(uids, sel) if s]
        wk = {u: float(val) for u, val, s in zip(uids, w[:, kk], sel) if s}
        matched = [u for u in uids_k if u in cs_row and u in pt_row]
        if not matched:
            print(f"      {labels[kk]}: 0 matched — skipping")
            continue
        cr = np.array([cs_row[u] for u in matched])
        pr = np.array([pt_row[u] for u in matched])
        new_scale = np.array([wk[u] for u in matched])

        pass_sub = pass_orig[pr].copy()
        if args.combine == "replace":
            pass_sub[args.scale_field] = new_scale.astype(scale_dtype.base)
        elif args.combine == "multiply":
            orig = pass_sub[args.scale_field].astype(np.float64)
            pass_sub[args.scale_field] = (orig * new_scale).astype(scale_dtype.base)

        cs_sub = cs_orig[cr].copy()
        base = f"{args.name_prefix}_{labels[kk]}"
        with open(out / f"{base}.cs", "wb") as fh:
            np.save(fh, cs_sub)
        with open(out / f"{base}_passthrough.cs", "wb") as fh:
            np.save(fh, pass_sub)
        manifest.append({
            "class": labels[kk], "n_particles": len(matched),
            "effective_N": float(new_scale.sum()),
            "mean_scale": float(new_scale.mean()),
            "particles_file": f"{base}.cs",
            "passthrough_file": f"{base}_passthrough.cs",
        })
        print(f"      {labels[kk]}: N={len(matched):,}  eff-N={new_scale.sum():.0f}  "
              f"-> {base}.cs (+ passthrough)")
    man_df = pd.DataFrame(manifest)
    man_df.to_csv(out / "export_manifest.csv", index=False)

    # ---- 5. summary -------------------------------------------------------
    print("[5/5] Writing SUMMARY.md")
    calib = "skipped" if args.no_calibrate else f"T*={T:.2f}"
    rows = "\n".join(
        f"| {m['class']} | {m['n_particles']:,} | {m['effective_N']:.0f} | "
        f"`{m['particles_file']}` + `{m['passthrough_file']}` |" for m in manifest)
    summary = f"""# Ensemble-GMM assignment + CryoSPARC export

This applies the GMM pipeline to the **combined** (ensemble) classifier and
turns the result into particle stacks you can reconstruct.

## What was done
1. Built the calibrated equal-weight ensemble posterior from the two experts:
   J1069 temperature-recalibrated ({calib}) to J1442's entropy, then linear pool
   at lambda={args.lam} ({N:,} shared protein particles).
2. Fit the GMM (k={k}, {args.covariance} cov, ALR embedding) to the ensemble
   posteriors. Converged={res.converged}, BIC={res.bic:.1f}, mean max-responsibility
   {max_resp.mean():.3f}.
3. Hard-assigned each particle by GMM argmax and exported disjoint per-class
   stacks, injecting the GMM responsibility as the per-particle scale
   ({args.scale_field}, combine={args.combine}, beta={args.beta}).

## Populations
{pop_df.round(4).to_string(index=False)}

Ensemble-argmax vs GMM-argmax agreement: {agree:.3f}.

## Exported stacks (import + "Homogeneous Reconstruction Only")
| class | N particles | effective N | files (in {out.as_posix()}) |
|---|---|---|---|
{rows}

### How to use in CryoSPARC
1. Import Particle Stack: give each `{args.name_prefix}_P?.cs` **and** its
   `_passthrough.cs` together.
2. Run **Homogeneous Reconstruction Only** with per-particle scale refinement
   **off** (the scale field already holds the ensemble-GMM weight). Use
   `--combine none` at export time instead if you want CryoSPARC's own weighting.
3. Compare the three maps as in `scripts/compare_maps.py`.

## Files
- ensemble_gmm_populations.csv, ensemble_gmm_assignment_uid.csv,
  ensemble_gmm_responsibilities.npy, export_manifest.csv
- {args.name_prefix}_P6/P7/P8.cs (+ _passthrough.cs)
"""
    (out / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(f"\nDone. Outputs in {out}")


if __name__ == "__main__":
    main()
