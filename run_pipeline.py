"""End-to-end runner for the GCER classification-uncertainty pipeline.

Usage
-----
    python run_pipeline.py --cs cryosparc_P25_J1442_00000_particles.cs \
                           --n-dummies 6 --outdir results_J1442

Pipeline steps (mapped to the project notes)
--------------------------------------------
0. Load CryoSPARC .cs file -> (N, K) posterior matrix.
1. Drop dummy classes, keep particles whose hard assignment is a protein class,
   renormalize their posteriors over the K_protein protein components only.
2. Transform posteriors out of the simplex (default: additive log-ratio ALR)
   so a Gaussian model is well posed.
3. Fit a GaussianMixture (n_components = K_protein), warm-started from the
   existing CryoSPARC hard assignments  -> Milestone 1.
4. Monte-Carlo estimate of the misclassification matrix C[i,j]
   = P(assigned=j | true=i), plus pairwise Bhattacharyya distances
   -> Milestone 2.
5. Population deconvolution pi_true = (C^T)^-1 pi_obs with bootstrap CIs
   -> Milestone 3.
6. (Optional) class-repetition analysis: refit with k+0, k+1, ... extra
   components and report occupancy migration  -> Milestone 4 / Hunt strategy 2.
7. Dump JSON + CSV + PNG diagnostics into --outdir.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from gmm_pipeline import (
    alr_transform,
    bhattacharyya_pairwise,
    bootstrap_population_ci,
    class_repetition_analysis,
    deconvolve_populations,
    fit_gmm,
    gmm_diagnostics,
    analytical_pairwise_confusion,
    hard_assignment_confusion,
    load_posteriors,
    monte_carlo_confusion,
    observed_populations,
)
from gmm_pipeline.plots import plot_confusion, plot_population_ci, plot_repetition


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cs", required=True, help="Path to CryoSPARC *_particles.cs file")
    p.add_argument("--n-dummies", type=int, default=6,
                   help="Number of leading dummy classes to drop")
    p.add_argument("--protein-idx", type=int, nargs="*", default=None,
                   help="Explicit 0-based indices of protein classes (overrides --n-dummies)")
    p.add_argument("--transform", choices=["alr", "drop"], default="alr",
                   help="Simplex embedding for Gaussian fitting")
    p.add_argument("--covariance", choices=["full", "diag", "tied", "spherical"], default="full")
    p.add_argument("--mc-samples", type=int, default=50_000,
                   help="Monte-Carlo samples per component for confusion matrix")
    p.add_argument("--n-boot", type=int, default=30, help="Bootstrap replicates")
    p.add_argument("--reps", type=int, nargs="*", default=[0, 1, 2, 3],
                   help="Extra components for class-repetition analysis")
    p.add_argument("--resp-threshold", type=float, default=0.9,
                   help="GMM max-responsibility cut for low-uncertainty export (default 0.9)")
    p.add_argument("--outdir", default="results", help="Output directory")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/8] Loading {args.cs}")
    post = load_posteriors(args.cs, protein_idx=args.protein_idx,
                           n_dummies=args.n_dummies)
    print(f"      N={len(post.uid):,}  K={post.n_classes}  "
          f"protein={list(post.protein_idx)}  dummy={list(post.dummy_idx)}")

    print("[2/8] Restricting to protein-assigned particles & renormalising")
    prot = post.protein_only()
    print(f"      N_protein={len(prot.uid):,}  K_protein={prot.n_protein}")
    pi_obs_raw = observed_populations(prot.hard_class, prot.n_protein)
    print(f"      observed populations (raw, hard-assignment): {np.round(pi_obs_raw,4)}")

    print(f"[3/8] Transform posteriors ({args.transform})")
    if args.transform == "alr":
        X = alr_transform(prot.posterior)
    else:
        from gmm_pipeline.preprocess import simplex_drop_last
        X = simplex_drop_last(prot.posterior)
    print(f"      X shape = {X.shape}")

    print(f"[4/8] Fitting GMM (n={prot.n_protein}, cov={args.covariance})")
    res = fit_gmm(X, n_components=prot.n_protein,
                  init_hard=prot.hard_class,
                  covariance_type=args.covariance,
                  random_state=args.seed)
    diag = gmm_diagnostics(res)
    print(f"      converged={diag['converged']}  iters={diag['n_iter']}  "
          f"BIC={diag['bic']:.1f}  AIC={diag['aic']:.1f}")
    pi_obs_gmm = observed_populations(res.hard_labels, prot.n_protein)
    print(f"      GMM-hard populations: {np.round(pi_obs_gmm,4)}")

    print(f"[5/8] Monte-Carlo confusion matrix ({args.mc_samples:,} samples/comp)")
    C = monte_carlo_confusion(res.model, args.mc_samples, random_state=args.seed)
    D_bhatt = bhattacharyya_pairwise(res.model)
    overlap = np.exp(-D_bhatt)
    np.fill_diagonal(overlap, 0.0)
    print(f"      diag(C) = {np.round(np.diag(C),4)}")
    print(f"      max off-diagonal overlap (Bhattacharyya coef.) = {overlap.max():.3f}")

    print("       Analytical pairwise confusion (Hunt erf formula, exact for K=2)")
    C_analytical = analytical_pairwise_confusion(prot.posterior, prot.hard_class)
    print(f"      diag(C_analytical) = {np.round(np.diag(C_analytical),4)}")

    M_empirical = hard_assignment_confusion(prot.hard_class, res.hard_labels, prot.n_protein)

    print("[6/8] Population deconvolution + bootstrap CIs")
    pi_corrected = deconvolve_populations(pi_obs_gmm, C)
    print(f"      corrected populations: {np.round(pi_corrected,4)}")

    def fit_fn(Xb):
        return GaussianMixture(
            n_components=prot.n_protein,
            covariance_type=args.covariance,
            reg_covar=1e-6, max_iter=500, tol=1e-5,
            random_state=args.seed, means_init=res.model.means_,
        ).fit(Xb)

    boot = bootstrap_population_ci(
        X, fit_fn, n_boot=args.n_boot,
        mc_samples=max(5_000, args.mc_samples // 5),
        random_state=args.seed,
    )
    print(f"      corrected mean +/- std: "
          f"{np.round(boot['corrected_mean'],4)} +/- {np.round(boot['corrected_std'],4)}")

    print("[7/8] Class-repetition analysis")
    rep = class_repetition_analysis(
        X, base_components=prot.n_protein, extra_range=args.reps,
        init_hard=prot.hard_class, covariance_type=args.covariance,
        random_state=args.seed,
    )
    print("      mapped weights vs r:")
    for r, w in zip(rep["r_values"], rep["mapped_weights"]):
        print(f"        r={r}: {np.round(w,4)}")

    # ----------------- persist -----------------
    labels = [f"P{int(c)}" for c in post.protein_idx]
    np.save(out / "posterior_protein.npy", prot.posterior)
    np.save(out / "responsibilities.npy", res.responsibilities)
    pd.DataFrame(C, index=labels, columns=labels).to_csv(out / "confusion_mc.csv")
    pd.DataFrame(C_analytical, index=labels, columns=labels).to_csv(out / "confusion_analytical.csv")
    pd.DataFrame(M_empirical, index=labels, columns=labels).to_csv(out / "confusion_empirical.csv")
    pd.DataFrame(overlap, index=labels, columns=labels).to_csv(out / "bhattacharyya_overlap.csv")
    pd.DataFrame({
        "class": labels,
        "observed": pi_obs_gmm,
        "corrected": pi_corrected,
        "corrected_mean_boot": boot["corrected_mean"],
        "corrected_std_boot": boot["corrected_std"],
        "corrected_lo": boot["corrected_lo"],
        "corrected_hi": boot["corrected_hi"],
    }).to_csv(out / "populations.csv", index=False)
    pd.DataFrame(rep["mapped_weights"],
                 index=[f"r={r}" for r in rep["r_values"]],
                 columns=labels).to_csv(out / "class_repetition.csv")

    with open(out / "gmm_diagnostics.json", "w") as f:
        json.dump({**diag, "protein_idx": list(map(int, post.protein_idx))}, f, indent=2)

    plot_confusion(C, labels, "Monte-Carlo confusion P(assigned|true)",
                   out / "confusion_mc.png")
    plot_confusion(C_analytical, labels,
                   "Analytical pairwise confusion P(classify j | true i)  [Hunt erf formula]",
                   out / "confusion_analytical.png")
    plot_confusion(M_empirical, labels, "Empirical CryoSPARC vs GMM hard agreement",
                   out / "confusion_empirical.png")
    plot_population_ci(pi_obs_gmm, boot["corrected_mean"],
                       boot["corrected_lo"], boot["corrected_hi"],
                       labels, out / "populations.png")
    plot_repetition(rep["r_values"], rep["mapped_weights"], labels,
                    out / "class_repetition.png")

    # ---- step 8: low-uncertainty particle export ----
    print(f"[8/8] Exporting low-uncertainty particles (max resp > {args.resp_threshold})")
    max_resp = res.responsibilities.max(axis=1)          # (N_protein,)
    gmm_hard_protein = res.hard_labels                   # 0-based within protein classes

    # max_resp is indexed into prot.uid (protein-only particle array)
    lo_mask = max_resp > args.resp_threshold
    lo_uids = prot.uid[lo_mask]
    lo_gmm_class = gmm_hard_protein[lo_mask]             # 0-based GMM component index
    lo_max_resp = max_resp[lo_mask]
    print(f"      {lo_mask.sum():,} / {len(max_resp):,} particles pass threshold "
          f"({100*lo_mask.mean():.1f} %)")
    for k in range(prot.n_protein):
        n_k = (lo_gmm_class == k).sum()
        print(f"        GMM component {k} ({labels[k]}): {n_k:,} particles")

    # Export a per-class subset .cs file
    cs_orig = np.load(args.cs)
    # Build a UID -> original-row index map
    uid_to_row = {int(uid): i for i, uid in enumerate(cs_orig["uid"])}
    for k in range(prot.n_protein):
        comp_mask = lo_gmm_class == k
        comp_uids = lo_uids[comp_mask]
        rows = np.array([uid_to_row[int(u)] for u in comp_uids if int(u) in uid_to_row])
        if len(rows) == 0:
            print(f"        {labels[k]}: no matching UIDs in original .cs — skipping")
            continue
        subset = cs_orig[rows]
        fname = out / f"low_uncertainty_{labels[k]}.cs"
        with open(fname, "wb") as fh:
            np.save(fh, subset)
        print(f"        {labels[k]}: saved {len(rows):,} particles -> {fname.name}")

    # Summary table
    pd.DataFrame({
        "uid": lo_uids,
        "gmm_component": lo_gmm_class,
        "gmm_label": [labels[k] for k in lo_gmm_class],
        "max_responsibility": lo_max_resp,
    }).to_csv(out / "low_uncertainty_particles.csv", index=False)
    print(f"      particle list -> low_uncertainty_particles.csv")

    print(f"\nDone. Outputs written to {out.resolve()}")


if __name__ == "__main__":
    main()
