"""3DVA latent-space analysis for the J1442 230k particle stack.

Two deliverables:
  1. Scatter / density plots of the three reaction-coordinate pairs
     (mode0 vs mode1, mode0 vs mode2, mode1 vs mode2) for ALL particles.
     Used to decide between:
        A  distinct blobs                  -> GMM appropriate
        B  continuous cloud                -> no discrete states
        C  continuous with density peaks   -> metastable regions on a
                                              fundamentally continuous landscape
  2. Fit a GMM directly to the 3DVA latent coordinates (mode0, mode1, mode2)
     -- the "3DVA-space GMM" -- and compare it to the existing
     "hetero-space GMM" that was fit to CryoSPARC class posteriors.

Run with the system python (numpy/sklearn/scipy/matplotlib/pandas all present):

    python scripts/analyze_3dva.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
LATENT_CS = (
    REPO
    / "data"
    / "J1442_3DVA"
    / "all_particles"
    / "components_mode_0"
    / "cryosparc_P25_J3428_particles.cs"
)
# hetero-space GMM artifacts (fit on CryoSPARC class posteriors, K=3)
HETERO_RESP = REPO / "results_J1442" / "gmm" / "responsibilities.npy"
J1442_CS = REPO / "data" / "cryosparc_P25_J1442_00000_particles.cs"
HETERO_PROTEIN_IDX = [6, 7, 8]          # P6, P7, P8 (dummy occupancy is 0%)
HETERO_LABELS = ["P6", "P7", "P8"]

OUTDIR = REPO / "results_J1442" / "threedva"
MODES = [0, 1, 2]
PAIRS = [(0, 1), (0, 2), (1, 2)]
RNG = 0


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_latents() -> tuple[np.ndarray, np.ndarray]:
    """Return (uid, X) where X is (N, 3) of 3DVA reaction coordinates."""
    cs = np.load(LATENT_CS)
    uid = np.asarray(cs["uid"]).astype(np.uint64)
    X = np.column_stack(
        [np.asarray(cs[f"components_mode_{m}/value"], dtype=np.float64) for m in MODES]
    )
    return uid, X


# --------------------------------------------------------------------------- #
# 1. Scatter / density plots
# --------------------------------------------------------------------------- #
def plot_density(X: np.ndarray) -> None:
    """Hexbin density (log counts) for the three mode pairs."""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    for ax, (i, j) in zip(axes, PAIRS):
        hb = ax.hexbin(
            X[:, i], X[:, j], gridsize=70, cmap="inferno", norm=LogNorm(), mincnt=1
        )
        ax.set_xlabel(f"mode {i}")
        ax.set_ylabel(f"mode {j}")
        ax.set_title(f"mode {i} vs mode {j}")
        fig.colorbar(hb, ax=ax, label="particles / bin (log)")
    fig.suptitle(
        f"J1442 3DVA latent density (all {len(X):,} particles)", fontsize=13
    )
    fig.tight_layout()
    fig.savefig(OUTDIR / "scatter_density_hexbin.png", dpi=150)
    plt.close(fig)


def plot_scatter(X: np.ndarray, max_pts: int = 40_000) -> None:
    """Raw subsampled scatter to read the cloud shape."""
    rng = np.random.default_rng(RNG)
    idx = (
        rng.choice(len(X), size=max_pts, replace=False)
        if len(X) > max_pts
        else np.arange(len(X))
    )
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    for ax, (i, j) in zip(axes, PAIRS):
        ax.scatter(X[idx, i], X[idx, j], s=2, alpha=0.08, color="steelblue", lw=0)
        ax.set_xlabel(f"mode {i}")
        ax.set_ylabel(f"mode {j}")
        ax.set_title(f"mode {i} vs mode {j}")
    fig.suptitle(
        f"J1442 3DVA latent scatter ({len(idx):,} of {len(X):,} particles)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(OUTDIR / "scatter_points.png", dpi=150)
    plt.close(fig)


def plot_marginals(X: np.ndarray) -> None:
    """1-D marginal histograms per mode (modality check)."""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.2))
    for ax, m in zip(axes, MODES):
        ax.hist(X[:, m], bins=120, color="slateblue", alpha=0.85)
        ax.set_xlabel(f"mode {m}")
        ax.set_ylabel("count")
        ax.set_title(
            f"mode {m}  (mean={X[:, m].mean():.2f}, sd={X[:, m].std():.2f})"
        )
    fig.suptitle("J1442 3DVA per-mode marginals", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / "mode_marginals.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 2a. BIC model selection (Scenario A / B / C evidence)
# --------------------------------------------------------------------------- #
def bic_sweep(Xs: np.ndarray, k_range=range(1, 11)) -> pd.DataFrame:
    rows = []
    for k in k_range:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            reg_covar=1e-6,
            max_iter=500,
            tol=1e-5,
            random_state=RNG,
            n_init=2,
        ).fit(Xs)
        rows.append(
            dict(
                k=k,
                bic=float(gmm.bic(Xs)),
                aic=float(gmm.aic(Xs)),
                loglik=float(gmm.score(Xs) * len(Xs)),
                converged=bool(gmm.converged_),
                min_weight=float(gmm.weights_.min()),
            )
        )
        print(f"  K={k:2d}  BIC={rows[-1]['bic']:.1f}  min_w={rows[-1]['min_weight']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "bic_sweep.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["k"], df["bic"], "o-", label="BIC")
    ax.plot(df["k"], df["aic"], "s--", color="gray", label="AIC", alpha=0.7)
    best = int(df.loc[df["bic"].idxmin(), "k"])
    ax.axvline(best, color="crimson", ls=":", label=f"min BIC @ K={best}")
    ax.set_xlabel("number of GMM components K")
    ax.set_ylabel("information criterion (lower = better)")
    ax.set_title("3DVA-space GMM model selection")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "bic_sweep.png", dpi=150)
    plt.close(fig)
    return df


def separation_stats(gmm: GaussianMixture) -> dict:
    """Standardised pairwise separation between component means:
    ||mu_a - mu_b|| / sqrt(0.5*(tr Sa + tr Sb)/d).  >~2 => well separated."""
    means = gmm.means_
    covs = gmm.covariances_
    d = means.shape[1]
    k = len(means)
    sep = np.zeros((k, k))
    for a in range(k):
        for b in range(k):
            if a == b:
                continue
            pooled = 0.5 * (np.trace(covs[a]) + np.trace(covs[b])) / d
            sep[a, b] = np.linalg.norm(means[a] - means[b]) / np.sqrt(pooled)
    off = sep[~np.eye(k, dtype=bool)]
    return {
        "separation_matrix": sep.tolist(),
        "min_separation_sd": float(off.min()),
        "mean_separation_sd": float(off.mean()),
    }


# --------------------------------------------------------------------------- #
# 2b. 3DVA-space GMM (K=3 to mirror the hetero-space GMM)
# --------------------------------------------------------------------------- #
def fit_3dva_gmm(Xs: np.ndarray, k: int) -> GaussianMixture:
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        reg_covar=1e-6,
        max_iter=500,
        tol=1e-5,
        random_state=RNG,
        n_init=4,
    ).fit(Xs)
    return gmm


def plot_gmm_assignment(X: np.ndarray, labels: np.ndarray, k: int) -> None:
    """Scatter of mode pairs colored by 3DVA-GMM hard label."""
    rng = np.random.default_rng(RNG)
    idx = (
        rng.choice(len(X), size=40_000, replace=False)
        if len(X) > 40_000
        else np.arange(len(X))
    )
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    for ax, (i, j) in zip(axes, PAIRS):
        for c in range(k):
            sel = idx[labels[idx] == c]
            ax.scatter(
                X[sel, i], X[sel, j], s=2, alpha=0.12, lw=0,
                color=cmap(c), label=f"comp {c}",
            )
        ax.set_xlabel(f"mode {i}")
        ax.set_ylabel(f"mode {j}")
        ax.set_title(f"mode {i} vs mode {j}")
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=cmap(c), label=f"comp {c}")
        for c in range(k)
    ]
    axes[-1].legend(handles=handles, markerscale=2, fontsize=8)
    fig.suptitle(f"3DVA-space GMM hard assignment (K={k})", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / "gmm3dva_assignment.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 3. Compare hetero-space GMM vs 3DVA-space GMM
# --------------------------------------------------------------------------- #
def hetero_hard_labels_by_uid() -> dict[int, int]:
    """Map uid -> hetero-space GMM hard component (argmax of saved resp).

    responsibilities.npy row i corresponds to row i of the J1442 stack
    (dummy occupancy 0% => protein_only kept all particles in order)."""
    resp = np.load(HETERO_RESP)              # (N, 3)
    cs = np.load(J1442_CS)
    uid = np.asarray(cs["uid"]).astype(np.uint64)
    assert len(uid) == len(resp), "hetero resp / J1442 uid length mismatch"
    hard = resp.argmax(axis=1)
    return dict(zip(uid.tolist(), hard.tolist()))


def compare_gmms(uid: np.ndarray, labels_3dva: np.ndarray, k: int) -> None:
    hetero = hetero_hard_labels_by_uid()
    keep = np.array([u in hetero for u in uid.tolist()])
    u_keep = uid[keep].tolist()
    lab3 = labels_3dva[keep]
    labh = np.array([hetero[u] for u in u_keep])
    n = len(u_keep)
    print(f"  matched {n:,} particles by uid for comparison")

    # contingency: rows = hetero (P6/P7/P8), cols = 3DVA comp 0..k-1
    cont = np.zeros((3, k), dtype=int)
    for a in range(3):
        for b in range(k):
            cont[a, b] = int(np.sum((labh == a) & (lab3 == b)))
    row_norm = cont / cont.sum(axis=1, keepdims=True).clip(min=1)

    pd.DataFrame(
        cont, index=HETERO_LABELS, columns=[f"3DVA_c{b}" for b in range(k)]
    ).to_csv(OUTDIR / "contingency_counts.csv")
    pd.DataFrame(
        row_norm, index=HETERO_LABELS, columns=[f"3DVA_c{b}" for b in range(k)]
    ).to_csv(OUTDIR / "contingency_rownorm.csv")

    # agreement metric independent of label permutation
    try:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        ari = float(adjusted_rand_score(labh, lab3))
        nmi = float(normalized_mutual_info_score(labh, lab3))
    except Exception:
        ari = nmi = float("nan")

    # heatmap
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    im = ax.imshow(row_norm, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(k))
    ax.set_xticklabels([f"3DVA c{b}" for b in range(k)])
    ax.set_yticks(range(3))
    ax.set_yticklabels(HETERO_LABELS)
    for a in range(3):
        for b in range(k):
            ax.text(
                b, a, f"{row_norm[a, b]:.2f}", ha="center", va="center",
                color="white" if row_norm[a, b] < 0.6 else "black", fontsize=9,
            )
    ax.set_title(
        f"hetero-space GMM (rows) vs 3DVA-space GMM (cols)\n"
        f"row-normalised  |  ARI={ari:.3f}  NMI={nmi:.3f}"
    )
    fig.colorbar(im, ax=ax, label="fraction of hetero class")
    fig.tight_layout()
    fig.savefig(OUTDIR / "gmm_comparison_heatmap.png", dpi=150)
    plt.close(fig)

    summary = {
        "n_matched": n,
        "adjusted_rand_index": ari,
        "normalized_mutual_info": nmi,
        "interpretation": (
            "ARI ~0 => the 3DVA latent partition is essentially unrelated to the "
            "CryoSPARC class-posterior partition; ARI ~1 => they agree."
        ),
    }
    (OUTDIR / "gmm_comparison_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  ARI={ari:.3f}  NMI={nmi:.3f}")


# --------------------------------------------------------------------------- #
def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    uid, X = load_latents()
    print(f"Loaded {len(X):,} particles, latent dim={X.shape[1]}")

    # ---- 1. plots ----
    print("Plotting density / scatter / marginals ...")
    plot_density(X)
    plot_scatter(X)
    plot_marginals(X)

    # standardise for GMM (modes have different variance scales)
    Xs = StandardScaler().fit_transform(X)

    # ---- 2a. BIC sweep ----
    print("BIC model-selection sweep (K=1..10) ...")
    bic_df = bic_sweep(Xs)
    best_k = int(bic_df.loc[bic_df["bic"].idxmin(), "k"])

    # ---- 2b. 3DVA-space GMM at K=3 (mirror hetero) and at best_k ----
    print("Fitting 3DVA-space GMM (K=3) ...")
    gmm3 = fit_3dva_gmm(Xs, k=3)
    sep = separation_stats(gmm3)
    diag = {
        "k": 3,
        "converged": bool(gmm3.converged_),
        "n_iter": int(gmm3.n_iter_),
        "bic": float(gmm3.bic(Xs)),
        "aic": float(gmm3.aic(Xs)),
        "weights": gmm3.weights_.tolist(),
        "means_standardized": gmm3.means_.tolist(),
        "component_cond_number": [float(np.linalg.cond(c)) for c in gmm3.covariances_],
        "best_k_by_bic": best_k,
        **sep,
    }
    (OUTDIR / "gmm3dva_diagnostics.json").write_text(json.dumps(diag, indent=2))
    print(
        f"  K=3 weights={np.round(gmm3.weights_,3)}  "
        f"min pairwise separation={sep['min_separation_sd']:.2f} SD  "
        f"(best K by BIC = {best_k})"
    )

    labels_3dva = gmm3.predict(Xs)
    plot_gmm_assignment(X, labels_3dva, k=3)

    # ---- 3. compare to hetero-space GMM ----
    print("Comparing hetero-space vs 3DVA-space GMM ...")
    compare_gmms(uid, labels_3dva, k=3)

    print(f"\nDone. Outputs in {OUTDIR}")


if __name__ == "__main__":
    main()
