"""Head-to-head GMM quality: 3DVA-latent space vs hetero-posterior space.

Both GMMs are fit with K=3 (P6/P7/P8 in the hetero case) on the SAME 230,396
particles, then scored with metrics that are comparable ACROSS the two spaces
(raw BIC is NOT comparable because the spaces / dimensions differ):

  * mean / median max responsibility  -> how confidently particles are assigned
  * fraction with max-resp > 0.8, > 0.9 -> size of the "confident" core
  * mean responsibility entropy (nats) -> 0 = crisp, ln(K)=1.10 = uniform mush
  * silhouette score (subsampled)      -> geometric cluster separation [-1, 1]
  * min / mean pairwise mean separation in pooled-SD units (>~2 = real clusters)

A GMM "looks better" (more discrete states) when assignments are confident,
entropy is low, silhouette is high, and separation is large.

Run with system python:  python scripts/compare_gmm_spaces.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import sys

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from gmm_pipeline.preprocess import alr_transform

LATENT_CS = (
    REPO / "data" / "J1442_3DVA" / "all_particles"
    / "components_mode_0" / "cryosparc_P25_J3428_particles.cs"
)
HETERO_POST = REPO / "results_J1442" / "gmm" / "posterior_protein.npy"
OUTDIR = REPO / "results_J1442" / "threedva"
K = 3
RNG = 0
SIL_SAMPLE = 20_000


def fit(X: np.ndarray, k: int) -> GaussianMixture:
    return GaussianMixture(
        n_components=k, covariance_type="full", reg_covar=1e-6,
        max_iter=500, tol=1e-5, random_state=RNG, n_init=4,
    ).fit(X)


def separation(gmm: GaussianMixture) -> tuple[float, float]:
    m, c = gmm.means_, gmm.covariances_
    d, k = m.shape[1], len(m)
    vals = []
    for a in range(k):
        for b in range(a + 1, k):
            pooled = 0.5 * (np.trace(c[a]) + np.trace(c[b])) / d
            vals.append(np.linalg.norm(m[a] - m[b]) / np.sqrt(pooled))
    return float(np.min(vals)), float(np.mean(vals))


def score(name: str, gmm: GaussianMixture, X: np.ndarray) -> dict:
    resp = gmm.predict_proba(X)
    hard = resp.argmax(axis=1)
    maxr = resp.max(axis=1)
    ent = -np.sum(np.where(resp > 0, resp * np.log(resp), 0.0), axis=1)
    rng = np.random.default_rng(RNG)
    idx = rng.choice(len(X), size=min(SIL_SAMPLE, len(X)), replace=False)
    sil = float(silhouette_score(X[idx], hard[idx]))
    smin, smean = separation(gmm)
    return {
        "space": name,
        "n": int(len(X)),
        "dim": int(X.shape[1]),
        "converged": bool(gmm.converged_),
        "mean_max_resp": float(maxr.mean()),
        "median_max_resp": float(np.median(maxr)),
        "frac_resp_gt_0.8": float((maxr > 0.8).mean()),
        "frac_resp_gt_0.9": float((maxr > 0.9).mean()),
        "mean_entropy_nats": float(ent.mean()),
        "max_entropy_uniform": float(np.log(K)),
        "silhouette": sil,
        "min_separation_sd": smin,
        "mean_separation_sd": smean,
        "weights": np.round(gmm.weights_, 4).tolist(),
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ---- hetero-posterior space (same transform as run_pipeline) ----
    post = np.load(HETERO_POST)            # (N, 3) protein posteriors
    Xh = alr_transform(post)               # (N, 2)
    gmm_h = fit(Xh, K)
    res_h = score("hetero_posterior", gmm_h, Xh)

    # ---- 3DVA latent space ----
    cs = np.load(LATENT_CS)
    Xl = np.column_stack(
        [np.asarray(cs[f"components_mode_{m}/value"], np.float64) for m in (0, 1, 2)]
    )
    Xl = StandardScaler().fit_transform(Xl)
    gmm_l = fit(Xl, K)
    res_l = score("3dva_latent", gmm_l, Xl)

    df = pd.DataFrame([res_h, res_l]).set_index("space")
    df.to_csv(OUTDIR / "gmm_quality_comparison.csv")
    (OUTDIR / "gmm_quality_comparison.json").write_text(
        json.dumps({"hetero_posterior": res_h, "3dva_latent": res_l}, indent=2)
    )
    print(df.T.to_string())

    # ---- figure: max-resp histograms + metric bars ----
    resp_h = gmm_h.predict_proba(Xh).max(1)
    resp_l = gmm_l.predict_proba(Xl).max(1)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    axes[0].hist(resp_h, bins=60, range=(1 / K, 1), alpha=0.6,
                 label="hetero posterior", color="tab:blue", density=True)
    axes[0].hist(resp_l, bins=60, range=(1 / K, 1), alpha=0.6,
                 label="3DVA latent", color="tab:orange", density=True)
    axes[0].axvline(1 / K, color="k", ls=":", label=f"chance (1/{K})")
    axes[0].set_xlabel("max responsibility per particle")
    axes[0].set_ylabel("density")
    axes[0].set_title("Assignment confidence\n(right-shifted = cleaner clusters)")
    axes[0].legend(fontsize=8)

    metrics = ["mean_max_resp", "frac_resp_gt_0.9", "silhouette",
               "min_separation_sd", "mean_separation_sd"]
    labels = ["mean\nmax-resp", "frac\n>0.9", "silhouette",
              "min sep\n(SD)", "mean sep\n(SD)"]
    hv = [res_h[m] for m in metrics]
    lv = [res_l[m] for m in metrics]
    x = np.arange(len(metrics))
    w = 0.38
    axes[1].bar(x - w / 2, hv, w, label="hetero posterior", color="tab:blue")
    axes[1].bar(x + w / 2, lv, w, label="3DVA latent", color="tab:orange")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_title("Higher = more discrete / better separated")
    axes[1].legend(fontsize=8)

    ent_h = -np.sum(np.where(gmm_h.predict_proba(Xh) > 0,
                             gmm_h.predict_proba(Xh) * np.log(gmm_h.predict_proba(Xh)), 0), 1)
    ent_l = -np.sum(np.where(gmm_l.predict_proba(Xl) > 0,
                             gmm_l.predict_proba(Xl) * np.log(gmm_l.predict_proba(Xl)), 0), 1)
    axes[2].hist(ent_h, bins=60, alpha=0.6, label="hetero posterior",
                 color="tab:blue", density=True)
    axes[2].hist(ent_l, bins=60, alpha=0.6, label="3DVA latent",
                 color="tab:orange", density=True)
    axes[2].axvline(np.log(K), color="k", ls=":", label=f"uniform = ln{K}={np.log(K):.2f}")
    axes[2].set_xlabel("responsibility entropy (nats)")
    axes[2].set_ylabel("density")
    axes[2].set_title("Lower = crisper assignment")
    axes[2].legend(fontsize=8)

    fig.suptitle("GMM quality: 3DVA-latent space vs hetero-posterior space (K=3)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / "gmm_quality_comparison.png", dpi=150)
    plt.close(fig)
    print(f"\nWrote {OUTDIR / 'gmm_quality_comparison.png'}")


if __name__ == "__main__":
    main()
