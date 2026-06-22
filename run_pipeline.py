"""End-to-end runner for the GCER classification-uncertainty pipeline.

Usage
-----
    python run_pipeline.py --cs cryosparc_P25_J1442_00000_particles.cs \
                           --n-dummies 6 --outdir results_J1442

See gmm_pipeline/README.md for full documentation and flag descriptions.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from gmm_pipeline import (
    alr_transform,
    bhattacharyya_pairwise,
    bootstrap_population_ci_analytical,
    bootstrap_gmm_parameters,
    class_repetition_analysis,
    deconvolve_populations,
    fit_gmm,
    gmm_diagnostics,
    gmm_confusion_equalprior,
    analytical_pairwise_confusion,
    analytical_multiclass_confusion,
    soft_posterior_confusion,
    hard_assignment_confusion,
    load_posteriors,
    monte_carlo_confusion,
    observed_populations,
)
from gmm_pipeline.plots import (
    plot_class_table,
    plot_confusion,
    plot_gmm_landscape,
    plot_population_ci,
    plot_population_comparison,
    plot_repetition,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cs", required=True, help="Path to CryoSPARC *_particles.cs file")
    p.add_argument("--passthrough-cs", default=None,
                   help="Optional path to matching CryoSPARC passthrough .cs file for low-uncertainty export")
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
    p.add_argument("--export-star", action="store_true",
                   help="Also export per-class .star files using pyem/csparc2star during low-uncertainty export")
    p.add_argument("--pyem-python", default=None,
                   help="Python executable with pyem installed (used by --export-star)")
    p.add_argument("--outdir", default="results", help="Output directory")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _convert_cs_to_star(cs_path: Path, passthrough_path: Path, star_path: Path,
                        pyem_python: str | None = None) -> str:
    """Convert CryoSPARC particle+passthrough .cs files to RELION .star.

    Tries common converter entry points in order and returns the successful
    command as a string. Raises RuntimeError if all methods fail.
    """
    attempts = []
    if pyem_python:
        attempts.append([pyem_python, "-m", "pyem.csparc2star",
                         str(cs_path), str(passthrough_path), str(star_path)])
    attempts.extend([
        ["csparc2star.py", str(cs_path), str(passthrough_path), str(star_path)],
        ["python", "-m", "pyem.csparc2star", str(cs_path), str(passthrough_path), str(star_path)],
        ["python3", "-m", "pyem.csparc2star", str(cs_path), str(passthrough_path), str(star_path)],
    ])

    errors = []
    for cmd in attempts:
        try:
            proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if proc.stdout.strip():
                print(f"          converter output: {proc.stdout.strip()}")
            return " ".join(cmd)
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            msg = str(exc)
            if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
                msg = f"{msg}; stderr={exc.stderr.strip()}"
            errors.append(f"{' '.join(cmd)} -> {msg}")

    joined = "\n".join(errors)
    raise RuntimeError(
        "Could not convert .cs to .star. Tried the following commands:\n" + joined
    )


def _sanitize_confusion(C: np.ndarray, n: int) -> np.ndarray:
    """Make a confusion matrix safe to invert: replace all-NaN rows (classes
    with too few particles to estimate) with the identity row so the matrix
    stays row-stochastic and non-singular."""
    C = np.array(C, dtype=np.float64, copy=True)
    for i in range(n):
        if not np.all(np.isfinite(C[i])) or C[i].sum() <= 0:
            C[i] = 0.0
            C[i, i] = 1.0
    return C


def main():
    args = parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    # Organised sub-folders (see gmm_pipeline/README.md "Outputs").
    conf_dir = out / "confusion"
    pop_dir = out / "populations"
    gmm_dir = out / "gmm"
    export_dir = out / "exports"
    for d in (conf_dir, pop_dir, gmm_dir, export_dir):
        d.mkdir(parents=True, exist_ok=True)

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
    C_analytical_pair = analytical_pairwise_confusion(prot.posterior, prot.hard_class)
    print(f"      diag(C_analytical_pair) = {np.round(np.diag(C_analytical_pair),4)}")

    print("       Analytical multi-class confusion (proper K>2 extension, score-space Gaussian)")
    C_analytical = analytical_multiclass_confusion(
        prot.posterior, prot.hard_class,
        n_samples=args.mc_samples, random_state=args.seed,
    )
    print(f"      diag(C_analytical_multi) = {np.round(np.diag(C_analytical),4)}")

    print("       GMM equal-prior confusion (geometry only, no mixing-weight bias)")
    C_gmm_eq = gmm_confusion_equalprior(res.model, args.mc_samples, random_state=args.seed)
    print(f"      diag(C_gmm_eq) = {np.round(np.diag(C_gmm_eq),4)}")

    print("       Soft-posterior confusion (honest, no selection bias, no GMM) -- primary")
    C_soft = soft_posterior_confusion(prot.posterior)
    print(f"      diag(C_soft) = {np.round(np.diag(C_soft),4)}")

    M_empirical = hard_assignment_confusion(prot.hard_class, res.hard_labels, prot.n_protein)

    pi_obs_soft = prot.posterior.mean(axis=0)
    print(f"      observed (soft, mean posterior over protein classes): {np.round(pi_obs_soft, 4)}")
    print(f"      observed (CryoSPARC-hard, argmax): {np.round(pi_obs_raw, 4)}")

    print("[6/8] Population deconvolution + bootstrap CIs")
    # Every confusion matrix, in reporting order (primary first). The flag marks
    # whether the matrix lives in CryoSPARC class space (honest, directly
    # invertible against pi_obs) or in GMM-component space (diagnostic; tagged
    # ".g" in tables/plots so the basis mismatch is explicit).
    confusion_set = {
        "soft": ("Soft-posterior", C_soft, True),
        "multi": ("Analytical-multi", C_analytical, True),
        "pair": ("Analytical-pair", C_analytical_pair, True),
        "mc.g": ("Monte-Carlo", C, False),
        "eq.g": ("GMM-equalprior", C_gmm_eq, False),
    }
    corrections, accuracies = {}, {}
    for key, (_disp, Cm, _honest) in confusion_set.items():
        Cm_clean = _sanitize_confusion(Cm, prot.n_protein)
        corrections[key] = deconvolve_populations(pi_obs_raw, Cm_clean)
        accuracies[key] = np.diag(Cm_clean)
        print(f"      corrected populations  [{key:5s}]: {np.round(corrections[key],4)}  "
              f"(diag acc {np.round(accuracies[key],3)})")
    pi_corrected_soft = corrections["soft"]
    pi_corrected_analytical = corrections["multi"]

    def fit_fn(Xb):
        return GaussianMixture(
            n_components=prot.n_protein,
            covariance_type=args.covariance,
            reg_covar=1e-6, max_iter=500, tol=1e-5,
            random_state=args.seed, means_init=res.model.means_,
        ).fit(Xb)

    boot_analytical = bootstrap_population_ci_analytical(
        prot.posterior, prot.hard_class, prot.n_protein,
        n_boot=max(args.n_boot, 200),
        mc_samples=max(5_000, args.mc_samples // 5),
        random_state=args.seed,
    )
    print(f"      corrected mean +/- std (analytic boot, no refit): "
          f"{np.round(boot_analytical['corrected_mean'],4)} +/- {np.round(boot_analytical['corrected_std'],4)}")

    print(f"      GMM parameter bootstrap ({args.n_boot} replicates, equal-prior confusion)...")
    boot_gmm = bootstrap_gmm_parameters(
        X, fit_fn, n_boot=args.n_boot,
        mc_samples=max(5_000, args.mc_samples // 5),
        random_state=args.seed,
    )
    print(f"      GMM means ||std|| per component: "
          f"{np.round(np.linalg.norm(boot_gmm['means_std'], axis=1), 4)}")
    print(f"      GMM confusion (equal-prior) bootstrap mean diag: "
          f"{np.round(np.diag(boot_gmm['confusion_mean']), 4)}")
    print(f"      GMM confusion (equal-prior) bootstrap std  diag: "
          f"{np.round(np.diag(boot_gmm['confusion_std']), 4)}")

    print("[7/8] Class-repetition analysis")
    rep = class_repetition_analysis(
        X, base_components=prot.n_protein, extra_range=args.reps,
        init_hard=prot.hard_class, covariance_type=args.covariance,
        random_state=args.seed,
    )
    print("      mapped weights vs r:")
    for r, w in zip(rep["r_values"], rep["mapped_weights"]):
        print(f"        r={r}: {np.round(w,4)}")

    labels = [f"P{int(c)}" for c in post.protein_idx]
    # ---- GMM model artifacts + raw arrays -> gmm/ ----
    np.save(gmm_dir / "posterior_protein.npy", prot.posterior)
    np.save(gmm_dir / "responsibilities.npy", res.responsibilities)
    pd.DataFrame(boot_gmm["means_mean"], index=labels).to_csv(gmm_dir / "gmm_means_mean.csv")
    pd.DataFrame(boot_gmm["means_std"], index=labels).to_csv(gmm_dir / "gmm_means_std.csv")
    pd.DataFrame(boot_gmm["weights_mean"], index=labels, columns=["weight_mean"]).to_csv(gmm_dir / "gmm_weights_mean.csv")
    pd.DataFrame(boot_gmm["weights_std"], index=labels, columns=["weight_std"]).to_csv(gmm_dir / "gmm_weights_std.csv")
    np.save(gmm_dir / "bootstrap_gmm_means.npy", boot_gmm["raw_means"])
    np.save(gmm_dir / "bootstrap_gmm_confusion.npy", boot_gmm["raw_confusion"])
    if "covs_mean" in boot_gmm:
        np.save(gmm_dir / "bootstrap_gmm_covs.npy", boot_gmm["raw_covs"])
    pd.DataFrame(rep["mapped_weights"],
                 index=[f"r={r}" for r in rep["r_values"]],
                 columns=labels).to_csv(gmm_dir / "gmm_class_repetition.csv")
    with open(gmm_dir / "gmm_diagnostics.json", "w") as f:
        json.dump({**diag, "protein_idx": list(map(int, post.protein_idx))}, f, indent=2)

    # ---- confusion matrices -> confusion/ ----
    pd.DataFrame(C, index=labels, columns=labels).to_csv(conf_dir / "confusion_montecarlo.csv")
    pd.DataFrame(C_gmm_eq, index=labels, columns=labels).to_csv(conf_dir / "confusion_gmm_equalprior.csv")
    pd.DataFrame(boot_gmm["confusion_mean"], index=labels, columns=labels).to_csv(conf_dir / "confusion_gmm_equalprior_mean.csv")
    pd.DataFrame(boot_gmm["confusion_std"], index=labels, columns=labels).to_csv(conf_dir / "confusion_gmm_equalprior_std.csv")
    pd.DataFrame(C_analytical, index=labels, columns=labels).to_csv(conf_dir / "confusion_multiclass_analytical.csv")
    pd.DataFrame(C_analytical_pair, index=labels, columns=labels).to_csv(conf_dir / "confusion_pairwise_analytical.csv")
    pd.DataFrame(C_soft, index=labels, columns=labels).to_csv(conf_dir / "confusion_soft_posterior.csv")
    pd.DataFrame(M_empirical, index=labels, columns=labels).to_csv(conf_dir / "confusion_empirical.csv")
    pd.DataFrame(overlap, index=labels, columns=labels).to_csv(conf_dir / "class_overlap_bhattacharyya.csv")

    # ---- populations -> populations/ ----
    pd.DataFrame({
        "class": labels,
        "observed_csparc_hard": pi_obs_raw,
        "observed_soft_mean_post": pi_obs_soft,
        "observed_gmm_hard": pi_obs_gmm,
        "corrected_soft_posterior": pi_corrected_soft,
        "corrected_analytical": pi_corrected_analytical,
        "corrected_mean_boot": boot_analytical["corrected_mean"],
        "corrected_std_boot": boot_analytical["corrected_std"],
        "corrected_lo": boot_analytical["corrected_lo"],
        "corrected_hi": boot_analytical["corrected_hi"],
    }).to_csv(pop_dir / "conformational_populations.csv", index=False)

    # Corrected populations + diagonal accuracy from EVERY confusion matrix,
    # so the effect of each inversion step is side by side.
    all_matrix_cols = {"class": labels, "observed_csparc_hard": pi_obs_raw}
    for key in confusion_set:
        all_matrix_cols[f"corrected_{key}"] = corrections[key]
        all_matrix_cols[f"accuracy_{key}"] = accuracies[key]
    pd.DataFrame(all_matrix_cols).to_csv(
        pop_dir / "population_corrections_all_matrices.csv", index=False)

    # ---- confusion plots -> confusion/ ----
    plot_confusion(C, labels, "Monte-Carlo confusion  (GMM-component space)",
                   conf_dir / "confusion_montecarlo.png")
    plot_confusion(C_gmm_eq, labels,
                   "GMM equal-prior confusion  (geometry only)",
                   conf_dir / "confusion_gmm_equalprior.png")
    plot_confusion(boot_gmm["confusion_mean"], labels,
                   "GMM equal-prior confusion  (bootstrap mean)",
                   conf_dir / "confusion_gmm_equalprior_mean.png")
    plot_confusion(boot_gmm["confusion_std"], labels,
                   "GMM equal-prior confusion  (bootstrap std)",
                   conf_dir / "confusion_gmm_equalprior_std.png")
    plot_confusion(C_analytical, labels,
                   "Analytical multi-class confusion",
                   conf_dir / "confusion_multiclass_analytical.png")
    plot_confusion(C_analytical_pair, labels,
                   "Analytical pairwise confusion",
                   conf_dir / "confusion_pairwise_analytical.png")
    plot_confusion(C_soft, labels,
                   "Soft-posterior confusion  (honest, primary)",
                   conf_dir / "confusion_soft_posterior.png")
    plot_confusion(M_empirical, labels, "Empirical CryoSPARC vs GMM hard agreement",
                   conf_dir / "confusion_empirical.png")
    plot_confusion(overlap, labels,
                   "Bhattacharyya overlap  (0 = distinct,  1 = identical)",
                   conf_dir / "class_overlap_bhattacharyya.png")

    # ---- population + GMM plots ----
    plot_population_ci(pi_obs_raw, pi_corrected_soft,
                       boot_analytical["corrected_std"],
                       labels=labels, out=pop_dir / "conformational_populations.png")
    plot_population_comparison(pi_obs_raw, corrections, labels=labels,
                               out=pop_dir / "population_corrections_all_matrices.png")
    plot_repetition(rep["r_values"], rep["mapped_weights"], labels,
                    gmm_dir / "gmm_class_repetition.png")
    plot_gmm_landscape(
        res.model, X, res.hard_labels, labels,
        title=f"Negative log-likelihood predicted by GMM  ({out.name})",
        out=gmm_dir / "gmm_nll_landscape.png",
    )
    plot_class_table(
        labels=labels,
        pi_obs=pi_obs_raw,
        corrections=corrections,
        accuracies=accuracies,
        primary="soft",
        pi_corr_std=boot_analytical["corrected_std"],
        pi_corr_lo=boot_analytical["corrected_lo"],
        pi_corr_hi=boot_analytical["corrected_hi"],
        extra_metrics={"Max\noverlap": overlap.max(axis=1)},
        title=f"{out.name}  ·  K={prot.n_protein}  ·  N={len(X):,}  ·  BIC={diag['bic']:.0f}",
        out=pop_dir / "summary_class_table.png",
    )

    print(f"[8/8] Exporting low-uncertainty particles (max resp > {args.resp_threshold})")
    max_resp = res.responsibilities.max(axis=1)
    gmm_hard_protein = res.hard_labels

    lo_mask = max_resp > args.resp_threshold
    lo_uids = prot.uid[lo_mask]
    lo_gmm_class = gmm_hard_protein[lo_mask]             # 0-based GMM component index
    lo_max_resp = max_resp[lo_mask]
    print(f"      {lo_mask.sum():,} / {len(max_resp):,} particles pass threshold "
          f"({100*lo_mask.mean():.1f} %)")
    for k in range(prot.n_protein):
        n_k = (lo_gmm_class == k).sum()
        print(f"        GMM component {k} ({labels[k]}): {n_k:,} particles")

    cs_orig = np.load(args.cs)
    passthrough_orig = None
    passthrough_uid_to_row = None
    if args.passthrough_cs:
        print(f"      loading passthrough source: {args.passthrough_cs}")
        passthrough_orig = np.load(args.passthrough_cs)
        if "uid" in passthrough_orig.dtype.names:
            passthrough_uid_to_row = {int(uid): i for i, uid in enumerate(passthrough_orig["uid"])}
        else:
            print("      warning: passthrough .cs has no uid field; falling back to particle-row indexing")
    uid_to_row = {int(uid): i for i, uid in enumerate(cs_orig["uid"])}
    for k in range(prot.n_protein):
        comp_mask = lo_gmm_class == k
        comp_uids = lo_uids[comp_mask]
        rows = np.array([uid_to_row[int(u)] for u in comp_uids if int(u) in uid_to_row])
        if len(rows) == 0:
            print(f"        {labels[k]}: no matching UIDs in original .cs — skipping")
            continue
        subset = cs_orig[rows]
        fname = export_dir / f"low_uncertainty_{labels[k]}.cs"
        with open(fname, "wb") as fh:
            np.save(fh, subset)
        print(f"        {labels[k]}: saved {len(rows):,} particles -> {fname.name}")

        passthrough_name = export_dir / f"low_uncertainty_{labels[k]}_passthrough.cs"
        if passthrough_orig is not None:
            if passthrough_uid_to_row is not None:
                pass_rows = np.array([
                    passthrough_uid_to_row[int(u)]
                    for u in comp_uids
                    if int(u) in passthrough_uid_to_row
                ])
            else:
                pass_rows = rows

            if len(pass_rows) == 0:
                print(f"        {labels[k]} passthrough: no matching rows found")
            else:
                with open(passthrough_name, "wb") as fh:
                    np.save(fh, passthrough_orig[pass_rows])
                print(f"        {labels[k]} passthrough: saved {len(pass_rows):,} rows -> {passthrough_name.name}")

                if args.export_star:
                    star_name = export_dir / f"low_uncertainty_{labels[k]}.star"
                    try:
                        cmd_used = _convert_cs_to_star(
                            fname, passthrough_name, star_name,
                            pyem_python=args.pyem_python,
                        )
                        print(f"        {labels[k]} star: saved -> {star_name.name} (via: {cmd_used})")
                    except RuntimeError as exc:
                        print(f"        {labels[k]} star: conversion failed\n{exc}")
        elif args.export_star:
            print(f"        {labels[k]} star: skipped (no --passthrough-cs provided)")

    pd.DataFrame({
        "uid": lo_uids,
        "gmm_component": lo_gmm_class,
        "gmm_label": [labels[k] for k in lo_gmm_class],
        "max_responsibility": lo_max_resp,
    }).to_csv(export_dir / "low_uncertainty_particles.csv", index=False)
    print(f"      particle list -> exports/low_uncertainty_particles.csv")

    print(f"\nDone. Outputs written to {out.resolve()}")


if __name__ == "__main__":
    main()
