"""Refit GMM from saved posteriors and compute equal-prior confusion + bootstrap CIs."""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gmm_pipeline import gmm_confusion_equalprior, bootstrap_gmm_parameters
from gmm_pipeline.preprocess import alr_transform
from gmm_pipeline.plots import plot_confusion

configs = [
    ("results_J1442", 3, [6, 7, 8]),
    ("results_J1497", 5, [6, 7, 8, 9, 10]),
]

for outdir, K, protein_idx in configs:
    print(f"\n=== {outdir}  K={K} ===")
    out = Path(outdir)
    conf_dir, gmm_dir = out / "confusion", out / "gmm"
    conf_dir.mkdir(parents=True, exist_ok=True)
    gmm_dir.mkdir(parents=True, exist_ok=True)
    labels = [f"P{c}" for c in protein_idx]

    # Load saved arrays
    post = np.load(gmm_dir / "posterior_protein.npy")   # (N, K)
    resp = np.load(gmm_dir / "responsibilities.npy")    # (N, K)
    X = alr_transform(post)                         # (N, K-1)

    # Warm-start means from saved responsibilities
    means_init = np.array([
        (resp[:, k : k + 1] * X).sum(0) / resp[:, k].sum()
        for k in range(K)
    ])

    gmm = GaussianMixture(
        n_components=K, covariance_type="full",
        reg_covar=1e-6, max_iter=500, tol=1e-5,
        random_state=0, means_init=means_init,
    ).fit(X)
    print(f"  converged={gmm.converged_}  iters={gmm.n_iter_}")
    print(f"  weights: {np.round(gmm.weights_, 4)}")

    # Point estimate: equal-prior confusion
    C_eq = gmm_confusion_equalprior(gmm, n_samples_per_component=50_000, random_state=0)
    print(f"  equal-prior diag: {np.round(np.diag(C_eq), 3)}")
    print("  full matrix:")
    print(np.round(C_eq, 3))

    # Bootstrap: means, covs, weights, confusion
    ref_means = gmm.means_.copy()

    def fit_fn(Xb):
        return GaussianMixture(
            n_components=K, covariance_type="full",
            reg_covar=1e-6, max_iter=500, tol=1e-5,
            random_state=0, means_init=ref_means,
        ).fit(Xb)

    print(f"  running bootstrap (n_boot=50)...")
    boot = bootstrap_gmm_parameters(
        X, fit_fn, n_boot=50, mc_samples=10_000, random_state=0
    )

    print(f"  confusion mean diag : {np.round(np.diag(boot['confusion_mean']), 3)}")
    print(f"  confusion std  diag : {np.round(np.diag(boot['confusion_std']),  3)}")
    print(f"  confusion 95% lo diag: {np.round(np.diag(boot['confusion_lo']),  3)}")
    print(f"  confusion 95% hi diag: {np.round(np.diag(boot['confusion_hi']),  3)}")
    print(f"  means ||std|| per component: {np.round(np.linalg.norm(boot['means_std'], axis=1), 4)}")
    print(f"  weights mean: {np.round(boot['weights_mean'], 4)}")
    print(f"  weights std : {np.round(boot['weights_std'],  4)}")

    # Save CSVs
    pd.DataFrame(C_eq, index=labels, columns=labels).to_csv(
        conf_dir / "confusion_gmm_equalprior.csv")
    pd.DataFrame(boot["confusion_mean"], index=labels, columns=labels).to_csv(
        conf_dir / "confusion_gmm_equalprior_mean.csv")
    pd.DataFrame(boot["confusion_std"], index=labels, columns=labels).to_csv(
        conf_dir / "confusion_gmm_equalprior_std.csv")
    pd.DataFrame(boot["confusion_lo"], index=labels, columns=labels).to_csv(
        conf_dir / "confusion_gmm_equalprior_lo.csv")
    pd.DataFrame(boot["confusion_hi"], index=labels, columns=labels).to_csv(
        conf_dir / "confusion_gmm_equalprior_hi.csv")
    pd.DataFrame(boot["means_mean"], index=labels).to_csv(gmm_dir / "gmm_means_mean.csv")
    pd.DataFrame(boot["means_std"],  index=labels).to_csv(gmm_dir / "gmm_means_std.csv")
    pd.DataFrame(boot["weights_mean"], index=labels, columns=["weight_mean"]).to_csv(
        gmm_dir / "gmm_weights_mean.csv")
    pd.DataFrame(boot["weights_std"], index=labels, columns=["weight_std"]).to_csv(
        gmm_dir / "gmm_weights_std.csv")
    np.save(gmm_dir / "bootstrap_gmm_means.npy",     boot["raw_means"])
    np.save(gmm_dir / "bootstrap_gmm_confusion.npy", boot["raw_confusion"])
    np.save(gmm_dir / "bootstrap_gmm_covs.npy",      boot["raw_covs"])

    # Plots
    plot_confusion(C_eq, labels,
                   f"GMM equal-prior confusion  ({outdir})",
                   conf_dir / "confusion_gmm_equalprior.png")
    plot_confusion(boot["confusion_mean"], labels,
                   f"GMM equal-prior — bootstrap mean  ({outdir})",
                   conf_dir / "confusion_gmm_equalprior_mean.png")
    plot_confusion(boot["confusion_std"], labels,
                   f"GMM equal-prior — bootstrap std  ({outdir})",
                   conf_dir / "confusion_gmm_equalprior_std.png")

print("\nDone.")
