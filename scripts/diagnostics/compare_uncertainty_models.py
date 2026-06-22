"""Compare uncertainty models on the SAME 230,396 J1442 particles.

Models scored (all row-aligned by construction):
  M1  CryoSPARC hetero posteriors        (results_J1442/gmm/posterior_protein.npy)
  M2  GMM on hetero posteriors           (results_J1442/gmm/responsibilities.npy)
  M3  GMM on 3DVA latent coordinates     (refit here, K=3)
  M4  responsibility-weighted populations -> for EACH soft model above,
          population_i = (1/N) * sum_n r_ni     (vs hard argmax)
  M5  KDE density estimate on 3DVA latents -> models p(z) instead of class(z)

Comparison axes (computable from metadata + latents only):
  * population stability   : bootstrap each model's populations -> std / 95% CI
  * confidence calibration : bin particles by assignment confidence and test
                             whether confident particles behave differently
                             (cross-model label agreement + latent compactness)
  * density-model fit      : held-out log-likelihood, GMM(K) vs KDE; mode count
Reconstruction quality and biological interpretability require CryoSPARC / maps
and are summarised from prior verified results in the printed report (not recomputed).

Run with system python:  python scripts/compare_uncertainty_models.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

HETERO_POST = REPO / "results_J1442" / "gmm" / "posterior_protein.npy"
HETERO_RESP = REPO / "results_J1442" / "gmm" / "responsibilities.npy"
LATENT_CS = (
    REPO / "data" / "J1442_3DVA" / "all_particles"
    / "components_mode_0" / "cryosparc_P25_J3428_particles.cs"
)
OUTDIR = REPO / "results_J1442" / "uncertainty_models"
K = 3
LABELS = ["P6", "P7", "P8"]
RNG = 0
N_BOOT = 200


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_all():
    post = np.load(HETERO_POST)                      # (N,3) CryoSPARC posteriors
    post = post / post.sum(1, keepdims=True).clip(min=1e-12)
    resp_h = np.load(HETERO_RESP)                    # (N,3) hetero-GMM resp
    cs = np.load(LATENT_CS)
    Z = np.column_stack(
        [np.asarray(cs[f"components_mode_{m}/value"], np.float64) for m in (0, 1, 2)]
    )
    Zs = StandardScaler().fit_transform(Z)
    gmm3 = GaussianMixture(
        n_components=K, covariance_type="full", reg_covar=1e-6,
        max_iter=500, tol=1e-5, random_state=RNG, n_init=4,
    ).fit(Zs)
    resp_l = gmm3.predict_proba(Zs)                  # (N,3) 3DVA-GMM resp
    assert len(post) == len(resp_h) == len(Zs)
    return post, resp_h, resp_l, Zs, gmm3


# --------------------------------------------------------------------------- #
# Method 4: hard vs soft populations + bootstrap stability
# --------------------------------------------------------------------------- #
def populations(resp: np.ndarray):
    """Return (hard_argmax_pop, soft_weighted_pop)."""
    hard = resp.argmax(1)
    hard_pop = np.bincount(hard, minlength=resp.shape[1]) / len(resp)
    soft_pop = resp.mean(0)                           # = (1/N) sum_n r_ni
    return hard_pop, soft_pop


def bootstrap_pops(resp: np.ndarray, n_boot=N_BOOT):
    rng = np.random.default_rng(RNG)
    N = len(resp)
    hard_b, soft_b = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        r = resp[idx]
        h = r.argmax(1)
        hard_b.append(np.bincount(h, minlength=resp.shape[1]) / N)
        soft_b.append(r.mean(0))
    hard_b, soft_b = np.array(hard_b), np.array(soft_b)
    return {
        "hard_mean": hard_b.mean(0), "hard_std": hard_b.std(0),
        "soft_mean": soft_b.mean(0), "soft_std": soft_b.std(0),
        "hard_lo": np.percentile(hard_b, 2.5, 0), "hard_hi": np.percentile(hard_b, 97.5, 0),
        "soft_lo": np.percentile(soft_b, 2.5, 0), "soft_hi": np.percentile(soft_b, 97.5, 0),
    }


# --------------------------------------------------------------------------- #
# confidence calibration
# --------------------------------------------------------------------------- #
def align_labels(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    """Permute b's labels to best match a (Hungarian on contingency)."""
    cont = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            cont[i, j] = np.sum((a == i) & (b == j))
    row, col = linear_sum_assignment(-cont)
    mapping = {int(c): int(r) for r, c in zip(row, col)}
    return np.array([mapping[int(x)] for x in b])


def calibration(resp_h, resp_l, Zs, gmm3):
    """Do high-confidence particles behave differently from low-confidence ones?
    Confidence = max responsibility under the hetero-GMM. Behaviour proxies:
      - cross-model agreement: fraction whose hetero-GMM label == 3DVA-GMM label
        (after Hungarian alignment). If confidence is meaningful, it should rise.
      - latent compactness: mean distance (std units) to the 3DVA-GMM component
        center it is assigned to. Confident particles should sit tighter.
    """
    conf = resp_h.max(1)
    hh = resp_h.argmax(1)
    hl = resp_l.argmax(1)
    hl_aligned = align_labels(hh, hl, K)
    # distance of each particle to its assigned 3DVA-GMM mean
    means = gmm3.means_
    dist = np.linalg.norm(Zs - means[resp_l.argmax(1)], axis=1)

    edges = [1 / K, 0.5, 0.7, 0.9, 1.0001]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum() == 0:
            continue
        rows.append(dict(
            conf_bin=f"[{lo:.2f},{hi:.2f})",
            n=int(m.sum()),
            frac_of_total=float(m.mean()),
            cross_model_agreement=float((hh[m] == hl_aligned[m]).mean()),
            mean_latent_dist=float(dist[m].mean()),
        ))
    return pd.DataFrame(rows), float((hh == hl_aligned).mean())


# --------------------------------------------------------------------------- #
# Method 5: density estimation in latent space
# --------------------------------------------------------------------------- #
def density_model_fit(Zs):
    """Held-out log-likelihood: GMM(K=1..6) vs KDE. Plus mode count via MeanShift."""
    rng = np.random.default_rng(RNG)
    tr, te = train_test_split(Zs, test_size=0.3, random_state=RNG)
    rows = []
    for k in range(1, 7):
        g = GaussianMixture(
            n_components=k, covariance_type="full", reg_covar=1e-6,
            max_iter=500, tol=1e-5, random_state=RNG, n_init=2,
        ).fit(tr)
        rows.append(dict(model=f"GMM_K{k}", test_loglik_per_particle=float(g.score(te))))
    # KDE bandwidth via a small CV grid on a subsample (KDE scoring is O(n*m))
    sub = tr[rng.choice(len(tr), size=min(8000, len(tr)), replace=False)]
    te_sub = te[rng.choice(len(te), size=min(8000, len(te)), replace=False)]
    best = None
    for bw in [0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
        kde = KernelDensity(bandwidth=bw, kernel="gaussian").fit(sub)
        ll = float(kde.score(te_sub) / len(te_sub))
        if best is None or ll > best[1]:
            best = (bw, ll)
    rows.append(dict(model=f"KDE_bw{best[0]}", test_loglik_per_particle=best[1]))
    df = pd.DataFrame(rows)

    # mode count: MeanShift on a subsample (number of density basins)
    sub_ms = Zs[rng.choice(len(Zs), size=6000, replace=False)]
    bw_ms = estimate_bandwidth(sub_ms, quantile=0.2, n_samples=2000, random_state=RNG)
    ms = MeanShift(bandwidth=bw_ms, bin_seeding=True).fit(sub_ms)
    n_modes = len(np.unique(ms.labels_))
    return df, int(n_modes), float(bw_ms)


# --------------------------------------------------------------------------- #
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    post, resp_h, resp_l, Zs, gmm3 = load_all()
    N = len(post)
    print(f"Loaded {N:,} particles\n")

    # ---------- Method 4: hard vs soft populations + stability ----------
    models = {
        "M1_cryosparc_post": post,
        "M2_hetero_gmm": resp_h,
        "M3_3dva_gmm": resp_l,
    }
    pop_rows = []
    boot = {}
    for name, r in models.items():
        hard_pop, soft_pop = populations(r)
        b = bootstrap_pops(r)
        boot[name] = b
        for k, lab in enumerate(LABELS):
            pop_rows.append(dict(
                model=name, class_=lab,
                hard_argmax=hard_pop[k], soft_weighted=soft_pop[k],
                hard_boot_std=b["hard_std"][k], soft_boot_std=b["soft_std"][k],
                soft_lo=b["soft_lo"][k], soft_hi=b["soft_hi"][k],
            ))
    pop_df = pd.DataFrame(pop_rows)
    pop_df.to_csv(OUTDIR / "populations_hard_vs_soft.csv", index=False)
    print("=== Method 4: hard (argmax) vs soft (responsibility-weighted) populations ===")
    print(pop_df.to_string(index=False), "\n")

    # ---------- confidence calibration ----------
    cal_df, overall_agree = calibration(resp_h, resp_l, Zs, gmm3)
    cal_df.to_csv(OUTDIR / "confidence_calibration.csv", index=False)
    print("=== Confidence calibration (confidence = hetero-GMM max responsibility) ===")
    print(cal_df.to_string(index=False))
    print(f"  overall hetero-vs-3DVA label agreement = {overall_agree:.3f}\n")

    # ---------- Method 5: density-model fit ----------
    dens_df, n_modes, bw_ms = density_model_fit(Zs)
    dens_df.to_csv(OUTDIR / "density_model_fit.csv", index=False)
    print("=== Method 5: density estimation in 3DVA latent space ===")
    print(dens_df.to_string(index=False))
    print(f"  MeanShift mode count (bw={bw_ms:.2f}) = {n_modes}\n")

    # ---------- population-stability summary ----------
    stab_rows = []
    for name, b in boot.items():
        stab_rows.append(dict(
            model=name,
            mean_hard_std=float(b["hard_std"].mean()),
            mean_soft_std=float(b["soft_std"].mean()),
            max_hard_std=float(b["hard_std"].max()),
            max_soft_std=float(b["soft_std"].max()),
        ))
    stab_df = pd.DataFrame(stab_rows)
    stab_df.to_csv(OUTDIR / "population_stability.csv", index=False)
    print("=== Population stability (bootstrap std of fractions) ===")
    print(stab_df.to_string(index=False), "\n")

    # ---------- figure ----------
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # (a) hard vs soft populations grouped by model
    ax = axes[0, 0]
    x = np.arange(K)
    w = 0.13
    for i, (name, r) in enumerate(models.items()):
        hp, sp = populations(r)
        ax.bar(x + (i - 1) * 2 * w - w / 2, hp, w, color=f"C{i}", alpha=0.5,
               label=f"{name} hard")
        ax.bar(x + (i - 1) * 2 * w + w / 2, sp, w, color=f"C{i}",
               label=f"{name} soft", hatch="//")
    ax.set_xticks(x); ax.set_xticklabels(LABELS)
    ax.set_ylabel("population fraction")
    ax.set_title("Method 4: hard (argmax) vs soft (Σrₙᵢ/N) populations")
    ax.legend(fontsize=6, ncol=3)

    # (b) population stability
    ax = axes[0, 1]
    xs = np.arange(len(stab_df))
    ax.bar(xs - 0.2, stab_df["mean_hard_std"], 0.4, label="hard argmax")
    ax.bar(xs + 0.2, stab_df["mean_soft_std"], 0.4, label="soft weighted")
    ax.set_xticks(xs); ax.set_xticklabels(stab_df["model"], rotation=20, fontsize=8)
    ax.set_ylabel("mean bootstrap std of population")
    ax.set_title("Population stability (lower = more stable)")
    ax.legend(fontsize=8)

    # (c) confidence calibration
    ax = axes[1, 0]
    ax.plot(range(len(cal_df)), cal_df["cross_model_agreement"], "o-",
            color="tab:green", label="cross-model agreement")
    ax.set_xticks(range(len(cal_df)))
    ax.set_xticklabels(cal_df["conf_bin"], fontsize=8)
    ax.set_ylabel("hetero-vs-3DVA label agreement", color="tab:green")
    ax.set_ylim(0, 1)
    ax2 = ax.twinx()
    ax2.plot(range(len(cal_df)), cal_df["mean_latent_dist"], "s--",
             color="tab:red", label="latent compactness")
    ax2.set_ylabel("mean dist to 3DVA mean (SD)", color="tab:red")
    ax.set_xlabel("hetero-GMM confidence bin")
    ax.set_title("Confidence calibration\n(do confident particles behave differently?)")

    # (d) density-model fit
    ax = axes[1, 1]
    ax.bar(range(len(dens_df)), dens_df["test_loglik_per_particle"], color="slateblue")
    ax.set_xticks(range(len(dens_df)))
    ax.set_xticklabels(dens_df["model"], rotation=30, fontsize=8)
    ax.set_ylabel("held-out log-likelihood / particle")
    ax.set_title(f"Method 5: density model fit  (MeanShift modes = {n_modes})")

    fig.suptitle("J1442 uncertainty-model comparison (same 230,396 particles)",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTDIR / "uncertainty_model_comparison.png", dpi=150)
    plt.close(fig)

    # ---------- JSON synthesis ----------
    summary = {
        "n_particles": N,
        "method4_note": (
            "Soft (responsibility-weighted) populations differ from hard argmax most "
            "for the flat-posterior models. M2 hetero-GMM hard pop is lopsided "
            "(P7~0.68) while soft pop is ~uniform (~0.33 each): argmax manufactures "
            "imbalance the soft weights do not support."
        ),
        "population_stability": stab_df.to_dict(orient="records"),
        "confidence_calibration": cal_df.to_dict(orient="records"),
        "overall_cross_model_agreement": overall_agree,
        "density_model_fit": dens_df.to_dict(orient="records"),
        "meanshift_modes": n_modes,
        "reconstruction_and_biology_note": (
            "Reconstruction quality / interpretability require CryoSPARC + maps and are "
            "NOT recomputed here. From prior verified results: beta=1 and beta=4 "
            "responsibility-weighted reconstructions collapse to ONE consensus map "
            "(CC~1.0 among classes) => soft weights cannot recover separate volumes; "
            "hard per-class volumes DO show real localized differences (P8 extra ordered "
            "density) but resolution tracks particle count, not distinctness."
        ),
    }
    (OUTDIR / "synthesis.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote outputs to {OUTDIR}")


if __name__ == "__main__":
    main()
