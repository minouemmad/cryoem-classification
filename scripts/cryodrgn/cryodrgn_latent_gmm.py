#!/usr/bin/env python
"""Latent-space GMM on a cryoDRGN embedding -> soft populations + uncertainty.

This is the lower half of the pipeline::

    CryoSPARC GCER (P6/P7/P8)
        -> cryoDRGN latent z            (already trained: z.N.pkl)
        -> latent-space GMM             (fit HERE, unsupervised)
        -> soft assignments             (responsibilities)
        -> bootstrap resampling         (particle bootstrap)
        -> population confidence intervals

It deliberately does NOT duplicate ``cryodrgn_compare.py`` (independent
KMeans/GMM clustering scored against references by ARI/NMI, latent scatter
coloured by CryoSPARC class). Run that first for the clustering-vs-reference
view; run THIS for:

  1. A GMM fit directly on the learned conformational manifold (the GMM moves
     from CryoSPARC class-posteriors to the cryoDRGN latent), with a BIC sweep
     to decide whether K discrete states are even supported.
  2. Soft populations + bootstrap CIs, and a bias-corrected population via the
     latent GMM's own soft-posterior confusion matrix (reusing gmm_pipeline).
  3. Per-particle CryoSPARC posterior vs. cryoDRGN-GMM posterior divergence
     (Jensen-Shannon -- the appropriate symmetric/bounded measure here -- plus
     KL both directions and a hard-agreement score), after Hungarian label
     alignment so the two 3-vectors live in the same {P6,P7,P8} basis.
  4. How separable the CryoSPARC classes are *inside* the latent (Bhattacharyya
     overlap of class-conditioned latent Gaussians) and how separable the latent
     GMM components are (standardised mean separation).
  5. A FAILURE-MODE diagnosis: BIC elbow, latent-GMM sharpness, component
     separation, silhouette, and the canonical correlation between z and the
     CryoSPARC posterior (does the latent even encode the classification axis?).

Alignment: z.N.pkl rows follow the order of the stack cryoDRGN was trained on
(the passthrough .cs). We map each z row to its CryoSPARC protein-only posterior
by uid, so the analysis is robust to any row-order differences between the
passthrough and the particles .cs.

Run with the cryoDRGN env (has numpy/sklearn/scipy/matplotlib) from repo root::

    python scripts/cryodrgn/cryodrgn_latent_gmm.py \
      --z results_cryodrgn/J1442_real/train/z.49.pkl \
      --passthrough-cs data/cryosparc_P25_J1442_passthrough_particles_all_classes.cs \
      --cs data/cryosparc_P25_J1442_00000_particles.cs \
      --n-dummies 6 -k 3 \
      --outdir results_cryodrgn/J1442_real/latent_gmm
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from gmm_pipeline.confusion import soft_posterior_confusion
from gmm_pipeline.data_io import load_posteriors
from gmm_pipeline.uncertainty import deconvolve_populations, observed_populations

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Loading / alignment
# --------------------------------------------------------------------------- #
def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_latent(z_path: str) -> np.ndarray:
    z = np.asarray(load_pkl(z_path), dtype=np.float64)
    if z.ndim == 1:
        z = z[:, None]
    print(f"[load] latent z {z.shape} from {z_path}")
    return z


def cs_uids(cs_path: str) -> np.ndarray:
    cs = np.load(cs_path)
    if "uid" not in (cs.dtype.names or ()):
        raise ValueError(f"{cs_path} has no uid field")
    return np.asarray(cs["uid"]).astype(np.uint64)


def align_z_to_posteriors(z, passthrough_cs, cs, n_dummies, protein_idx):
    """Return (z_a, cryo_post_a, cryo_hard_a, uid_a) aligned by uid.

    z rows follow the passthrough stack order; we map each to the CryoSPARC
    protein-only posterior for the same particle.
    """
    uid_pass = cs_uids(passthrough_cs)
    if len(uid_pass) != len(z):
        print(f"[align] WARNING: passthrough uids ({len(uid_pass)}) != z rows "
              f"({len(z)}); using min and assuming leading correspondence.")
        m = min(len(uid_pass), len(z))
        uid_pass, z = uid_pass[:m], z[:m]

    post = load_posteriors(cs, protein_idx=protein_idx, n_dummies=n_dummies)
    prot = post.protein_only()
    uid_prot = np.asarray(prot.uid).astype(np.uint64)
    # uid -> row in prot (keep as Python int keys; uids exceed 2**53)
    row_of = {int(u): i for i, u in enumerate(uid_prot.tolist())}

    keep_z, keep_rows = [], []
    for i, u in enumerate(uid_pass.tolist()):
        r = row_of.get(int(u))
        if r is not None:
            keep_z.append(i)
            keep_rows.append(r)
    keep_z = np.asarray(keep_z)
    keep_rows = np.asarray(keep_rows)
    print(f"[align] matched {len(keep_z):,} / {len(z):,} latent rows to "
          f"protein-only CryoSPARC posteriors")

    z_a = z[keep_z]
    cryo_post_a = prot.posterior[keep_rows]
    cryo_hard_a = prot.hard_class[keep_rows]
    uid_a = uid_pass[keep_z]
    return z_a, cryo_post_a, cryo_hard_a, uid_a, prot.n_protein


# --------------------------------------------------------------------------- #
# Latent GMM + model selection
# --------------------------------------------------------------------------- #
def bic_sweep(Xs, k_range, seed, outdir):
    rows = []
    for k in k_range:
        g = GaussianMixture(k, covariance_type="full", reg_covar=1e-6,
                            max_iter=500, tol=1e-5, n_init=3,
                            random_state=seed).fit(Xs)
        rows.append({"k": k, "bic": float(g.bic(Xs)), "aic": float(g.aic(Xs)),
                     "loglik": float(g.score(Xs) * len(Xs)),
                     "min_weight": float(g.weights_.min()),
                     "converged": bool(g.converged_)})
        print(f"  K={k:2d}  BIC={rows[-1]['bic']:.1f}  min_w={rows[-1]['min_weight']:.4f}")
    ks = [r["k"] for r in rows]
    bics = [r["bic"] for r in rows]
    aics = [r["aic"] for r in rows]
    best = ks[int(np.argmin(bics))]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, bics, "o-", label="BIC")
    ax.plot(ks, aics, "s--", color="gray", alpha=0.7, label="AIC")
    ax.axvline(best, color="crimson", ls=":", label=f"min BIC @ K={best}")
    ax.set_xlabel("number of latent-GMM components K")
    ax.set_ylabel("information criterion (lower = better)")
    ax.set_title("cryoDRGN latent-space GMM model selection")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "latent_bic_sweep.png"), dpi=150)
    plt.close(fig)
    return rows, best


def separation_stats(gmm: GaussianMixture) -> dict:
    """Standardised pairwise separation ||mu_a-mu_b|| / sqrt(pooled var).
    >~2 => well separated components."""
    means, covs = gmm.means_, gmm.covariances_
    d, k = means.shape[1], len(means)
    sep = np.zeros((k, k))
    for a in range(k):
        for b in range(k):
            if a == b:
                continue
            pooled = 0.5 * (np.trace(covs[a]) + np.trace(covs[b])) / d
            sep[a, b] = np.linalg.norm(means[a] - means[b]) / np.sqrt(max(pooled, EPS))
    off = sep[~np.eye(k, dtype=bool)]
    return {"separation_matrix": sep.tolist(),
            "min_separation_sd": float(off.min()),
            "mean_separation_sd": float(off.mean())}


# --------------------------------------------------------------------------- #
# Label alignment (latent GMM components -> CryoSPARC classes)
# --------------------------------------------------------------------------- #
def align_components(cryo_post, drgn_resp):
    """Hungarian alignment of latent-GMM columns onto CryoSPARC class columns
    using the soft cross-tabulation T[i,j] = sum_n cryo[n,i] * drgn[n,j].
    Returns the column permutation perm such that drgn[:, perm] matches cryo."""
    T = cryo_post.T @ drgn_resp                      # (K, K)
    rows, cols = linear_sum_assignment(-T)
    perm = np.empty(T.shape[1], dtype=int)
    perm[rows] = cols
    return perm, T


# --------------------------------------------------------------------------- #
# Per-particle divergence
# --------------------------------------------------------------------------- #
def _safe(p):
    p = np.clip(p, EPS, None)
    return p / p.sum(axis=1, keepdims=True)


def kl_rows(p, q):
    p, q = _safe(p), _safe(q)
    return np.sum(p * np.log(p / q), axis=1)


def js_rows(p, q):
    """Jensen-Shannon divergence per row (nats; bounded by ln2)."""
    p, q = _safe(p), _safe(q)
    m = 0.5 * (p + q)
    return 0.5 * np.sum(p * np.log(p / m), axis=1) + 0.5 * np.sum(q * np.log(q / m), axis=1)


# --------------------------------------------------------------------------- #
# Latent class overlap (Bhattacharyya of class-conditioned Gaussians)
# --------------------------------------------------------------------------- #
def bhattacharyya(m1, S1, m2, S2):
    S = 0.5 * (S1 + S2)
    d = m1 - m2
    Si = np.linalg.pinv(S)
    term1 = 0.125 * d @ Si @ d
    s1 = max(np.linalg.det(S1), EPS)
    s2 = max(np.linalg.det(S2), EPS)
    sdet = max(np.linalg.det(S), EPS)
    term2 = 0.5 * np.log(sdet / np.sqrt(s1 * s2))
    return float(term1 + term2)


def class_overlap_in_latent(z, cryo_hard, k, labels):
    means, covs = [], []
    for c in range(k):
        zc = z[cryo_hard == c]
        if len(zc) < z.shape[1] + 2:
            means.append(None); covs.append(None); continue
        means.append(zc.mean(0))
        covs.append(np.cov(zc, rowvar=False) + 1e-6 * np.eye(z.shape[1]))
    ov = np.zeros((k, k))
    for a in range(k):
        for b in range(k):
            if a == b or means[a] is None or means[b] is None:
                continue
            ov[a, b] = np.exp(-bhattacharyya(means[a], covs[a], means[b], covs[b]))
    return ov


# --------------------------------------------------------------------------- #
# Canonical correlation z <-> CryoSPARC posterior (failure diagnostic)
# --------------------------------------------------------------------------- #
def canonical_correlations(z, cryo_post):
    """First few canonical correlations between standardized z and ALR of the
    CryoSPARC posterior. Low values => the latent does not linearly encode the
    classification axis (a concrete cryoDRGN failure signature)."""
    X = StandardScaler().fit_transform(z)
    p = _safe(cryo_post)
    Y = np.log(p[:, :-1]) - np.log(p[:, -1:])         # ALR
    Y = StandardScaler().fit_transform(Y)
    n = len(X)
    Cxx = X.T @ X / n + 1e-4 * np.eye(X.shape[1])
    Cyy = Y.T @ Y / n + 1e-4 * np.eye(Y.shape[1])
    Cxy = X.T @ Y / n
    ix = np.linalg.inv(np.linalg.cholesky(Cxx))
    iy = np.linalg.inv(np.linalg.cholesky(Cyy))
    M = ix @ Cxy @ iy.T
    s = np.linalg.svd(M, compute_uv=False)
    return np.clip(s, 0, 1)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def heatmap(M, row_labels, col_labels, title, path, fmt="{:.2f}", vmax=None):
    fig, ax = plt.subplots(figsize=(1.5 + 0.9 * M.shape[1], 1.5 + 0.8 * M.shape[0]))
    im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=0,
                   vmax=vmax if vmax is not None else M.max())
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(col_labels)
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(row_labels)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                    color="white" if M[i, j] < 0.6 * (vmax or M.max() or 1) else "black",
                    fontsize=9)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def population_plot(labels, soft, soft_lo, soft_hi, corrected, corr_lo, corr_hi,
                    cryo_soft, path):
    x = np.arange(len(labels))
    w = 0.27
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(x - w, cryo_soft, w, label="CryoSPARC soft", color="#9aa7b1")
    ax.bar(x, soft, w, yerr=[soft - soft_lo, soft_hi - soft],
           capsize=4, label="latent-GMM soft", color="#3b7dd8")
    ax.bar(x + w, corrected, w, yerr=[corrected - corr_lo, corr_hi - corrected],
           capsize=4, label="latent-GMM corrected", color="#d9822b")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("population fraction")
    ax.set_title("Conformational populations (95% bootstrap CI)")
    ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def divergence_plot(js, agree, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(js, bins=80, color="#6a51a3")
    axes[0].axvline(js.mean(), color="crimson", ls=":",
                    label=f"mean={js.mean():.3f} nats")
    axes[0].set_xlabel("Jensen-Shannon divergence (nats, bound ln2=0.693)")
    axes[0].set_ylabel("particles"); axes[0].set_title("CryoSPARC vs cryoDRGN-GMM posterior")
    axes[0].legend()
    axes[1].bar([0, 1], [1 - agree.mean(), agree.mean()],
                color=["#bdbdbd", "#31a354"])
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["disagree", "agree"])
    axes[1].set_ylabel("fraction")
    axes[1].set_title(f"Hard-label agreement = {agree.mean():.3f}")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--z", required=True, help="cryoDRGN z.N.pkl latent embedding.")
    p.add_argument("--passthrough-cs", required=True,
                   help="CryoSPARC .cs the cryoDRGN stack was trained on (uid order of z).")
    p.add_argument("--cs", required=True,
                   help="CryoSPARC particles .cs with class posteriors.")
    p.add_argument("--n-dummies", type=int, default=6)
    p.add_argument("--protein-idx", type=int, nargs="*", default=None)
    p.add_argument("-k", "--k", type=int, default=None,
                   help="Latent-GMM components (default: #protein classes).")
    p.add_argument("--k-max", type=int, default=8, help="Max K for BIC sweep.")
    p.add_argument("--n-boot", type=int, default=500, help="Particle bootstrap replicates.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-o", "--outdir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    z = load_latent(args.z)
    z, cryo_post, cryo_hard, uid, n_protein = align_z_to_posteriors(
        z, args.passthrough_cs, args.cs, args.n_dummies, args.protein_idx)
    k = args.k or n_protein
    labels = [f"P{6 + i}" for i in range(k)] if k == n_protein else [f"S{i}" for i in range(k)]

    Xs = StandardScaler().fit_transform(z)

    # 1. model selection -------------------------------------------------------
    print("[1] BIC sweep on standardized latent")
    sweep, best_k = bic_sweep(Xs, range(1, args.k_max + 1), args.seed, args.outdir)

    # 2. latent GMM (unsupervised) at K -----------------------------------------
    print(f"[2] Fitting latent-space GMM K={k} (unsupervised)")
    gmm = GaussianMixture(k, covariance_type="full", reg_covar=1e-6, max_iter=1000,
                          tol=1e-6, n_init=10, random_state=args.seed).fit(Xs)
    drgn_resp = gmm.predict_proba(Xs)                       # (N, k)
    drgn_hard = drgn_resp.argmax(axis=1)
    sep = separation_stats(gmm)
    sil = float(silhouette_score(Xs, drgn_hard, sample_size=min(10000, len(Xs)),
                                 random_state=args.seed)) if k > 1 else float("nan")

    # 3. align components to CryoSPARC classes ---------------------------------
    perm, cross = align_components(cryo_post, drgn_resp)
    drgn_resp = drgn_resp[:, perm]
    drgn_hard = drgn_resp.argmax(axis=1)
    print(f"[3] component->class permutation: {perm.tolist()}")

    # 4. populations + bootstrap CIs (the pipeline payoff) ---------------------
    print(f"[4] Populations + particle bootstrap (n_boot={args.n_boot})")
    pi_obs_hard = observed_populations(cryo_hard, k)        # CryoSPARC hard counts
    soft_pop = drgn_resp.mean(axis=0)                       # latent-GMM soft population
    cryo_soft = cryo_post.mean(axis=0)
    C_soft = soft_posterior_confusion(drgn_resp)
    corrected = deconvolve_populations(pi_obs_hard, C_soft)

    N = len(z)
    soft_boot, corr_boot = [], []
    for _ in range(args.n_boot):
        idx = rng.integers(0, N, size=N)
        rb = drgn_resp[idx]
        soft_boot.append(rb.mean(axis=0))
        pi_b = observed_populations(cryo_hard[idx], k)
        corr_boot.append(deconvolve_populations(pi_b, soft_posterior_confusion(rb)))
    soft_boot = np.array(soft_boot)
    corr_boot = np.array(corr_boot)
    soft_lo, soft_hi = np.percentile(soft_boot, [2.5, 97.5], axis=0)
    corr_lo, corr_hi = np.percentile(corr_boot, [2.5, 97.5], axis=0)

    # 5. per-particle divergence -----------------------------------------------
    print("[5] Per-particle CryoSPARC vs cryoDRGN-GMM divergence")
    js = js_rows(cryo_post, drgn_resp)
    kl_cd = kl_rows(cryo_post, drgn_resp)
    kl_dc = kl_rows(drgn_resp, cryo_post)
    agree = (cryo_post.argmax(1) == drgn_resp.argmax(1)).astype(float)

    # 6. overlaps --------------------------------------------------------------
    print("[6] Latent class overlap (Bhattacharyya) + component separation")
    cls_overlap = class_overlap_in_latent(z, cryo_hard, k, labels)

    # 7. failure diagnostics ---------------------------------------------------
    print("[7] Failure-mode diagnostics")
    ccs = canonical_correlations(z, cryo_post)
    bic1 = next(r["bic"] for r in sweep if r["k"] == 1)
    bick = next((r["bic"] for r in sweep if r["k"] == k), float("nan"))
    delta_bic = bic1 - bick                                  # >0 => K beats K=1
    mean_maxresp = float(drgn_resp.max(axis=1).mean())
    frac_conf = float((drgn_resp.max(axis=1) > 0.9).mean())

    # ---- plots ----
    population_plot(labels, soft_pop, soft_lo, soft_hi, corrected, corr_lo, corr_hi,
                    cryo_soft, os.path.join(args.outdir, "populations_ci.png"))
    divergence_plot(js, agree, os.path.join(args.outdir, "per_particle_divergence.png"))
    heatmap(C_soft, labels, labels, "Latent-GMM soft-posterior confusion",
            os.path.join(args.outdir, "latent_confusion_soft.png"), vmax=1.0)
    if k > 1:
        heatmap(cls_overlap, labels, labels,
                "CryoSPARC class overlap in latent (Bhattacharyya)",
                os.path.join(args.outdir, "latent_class_overlap.png"), vmax=1.0)
        heatmap(np.array(sep["separation_matrix"]), labels, labels,
                "Latent-GMM component separation (SD units)",
                os.path.join(args.outdir, "latent_component_separation.png"),
                fmt="{:.2f}", vmax=None)

    # ---- numeric outputs ----
    np.savez(os.path.join(args.outdir, "per_particle.npz"),
             uid=uid.astype(np.uint64), cryosparc_posterior=cryo_post,
             cryodrgn_gmm_posterior=drgn_resp, js_divergence=js,
             kl_cryo_drgn=kl_cd, kl_drgn_cryo=kl_dc, agreement=agree)

    import csv
    with open(os.path.join(args.outdir, "populations.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "cryosparc_soft", "cryosparc_hard",
                    "latentgmm_soft", "soft_lo", "soft_hi",
                    "latentgmm_corrected", "corr_lo", "corr_hi"])
        for i, lab in enumerate(labels):
            w.writerow([lab, f"{cryo_soft[i]:.4f}", f"{pi_obs_hard[i]:.4f}",
                        f"{soft_pop[i]:.4f}", f"{soft_lo[i]:.4f}", f"{soft_hi[i]:.4f}",
                        f"{corrected[i]:.4f}", f"{corr_lo[i]:.4f}", f"{corr_hi[i]:.4f}"])

    diagnosis = {
        "n_particles": int(N),
        "zdim": int(z.shape[1]),
        "k": int(k),
        "bic_best_k": int(best_k),
        "delta_bic_k_vs_1": float(delta_bic),
        "latent_gmm_mean_max_responsibility": mean_maxresp,
        "latent_gmm_frac_resp_gt_0.9": frac_conf,
        "latent_gmm_min_separation_sd": sep["min_separation_sd"],
        "latent_gmm_mean_separation_sd": sep["mean_separation_sd"],
        "latent_gmm_silhouette": sil,
        "canonical_correlations_z_vs_cryosparc": ccs.tolist(),
        "max_canonical_correlation": float(ccs.max()),
        "hard_agreement_cryosparc_vs_latentgmm": float(agree.mean()),
        "mean_js_divergence_nats": float(js.mean()),
        "median_js_divergence_nats": float(np.median(js)),
        "component_to_class_permutation": perm.tolist(),
    }
    with open(os.path.join(args.outdir, "diagnostics.json"), "w") as f:
        json.dump(diagnosis, f, indent=2)

    # ---- failure verdict ----
    flags = []
    if best_k == 1 or delta_bic < 0.005 * abs(bic1):
        flags.append("BIC barely prefers K>1 -> latent is ~unimodal (continuous "
                     "landscape; discrete K states not supported).")
    if sep["min_separation_sd"] < 2.0:
        flags.append(f"min component separation {sep['min_separation_sd']:.2f} SD "
                     "(<2) -> latent GMM components are arbitrary slices of one cloud.")
    if mean_maxresp < 0.7:
        flags.append(f"mean max responsibility {mean_maxresp:.2f} -> soft, "
                     "overlapping latent assignments.")
    if ccs.max() < 0.3:
        flags.append(f"max canonical corr {ccs.max():.2f} -> the latent does NOT "
                     "linearly encode the CryoSPARC classification axis (cryoDRGN "
                     "learned a different/again continuous coordinate, or undertrained).")
    if agree.mean() < 0.5:
        flags.append(f"hard agreement {agree.mean():.2f} -> latent partition "
                     "largely disagrees with CryoSPARC classes.")
    if not flags:
        flags.append("Latent GMM shows separated, confident, CryoSPARC-consistent "
                     "states: discrete K-state model is supported here.")

    summary = [
        "# cryoDRGN latent-space GMM: populations, divergence, failure diagnosis",
        "",
        f"- particles matched: {N:,}  |  zdim: {z.shape[1]}  |  K: {k}",
        f"- BIC-best K (1..{args.k_max}): {best_k}   (dBIC K={k} vs K=1: {delta_bic:.1f})",
        f"- latent-GMM mean max-resp: {mean_maxresp:.3f}  (frac>0.9: {frac_conf:.3f})",
        f"- component separation (SD): min {sep['min_separation_sd']:.2f}, "
        f"mean {sep['mean_separation_sd']:.2f}  |  silhouette {sil:.3f}",
        f"- canonical corr z vs CryoSPARC: {np.round(ccs, 3).tolist()} (max {ccs.max():.3f})",
        f"- hard agreement CryoSPARC vs latent-GMM: {agree.mean():.3f}",
        f"- mean / median JS divergence: {js.mean():.3f} / {np.median(js):.3f} nats "
        f"(bound ln2 = 0.693)",
        "",
        "## Populations (95% bootstrap CI)",
        "",
        "| class | CryoSPARC soft | latent-GMM soft | latent-GMM corrected |",
        "|---|---|---|---|",
    ]
    for i, lab in enumerate(labels):
        summary.append(
            f"| {lab} | {cryo_soft[i]:.3f} | "
            f"{soft_pop[i]:.3f} [{soft_lo[i]:.3f}, {soft_hi[i]:.3f}] | "
            f"{corrected[i]:.3f} [{corr_lo[i]:.3f}, {corr_hi[i]:.3f}] |")
    summary += ["", "## Failure-mode verdict", ""]
    summary += [f"- {f}" for f in flags]
    summary += [
        "",
        "## Suggested next steps",
        "",
        "- If the latent is unimodal / low canonical-corr: this corroborates the "
        "flat-posterior finding — report a continuous reaction coordinate "
        "(pc1/UMAP traversal) and populations along it, not discrete fractions.",
        "- To stress-test cryoDRGN itself before concluding: retrain at higher "
        "`--zdim` (e.g. 10) and/or more epochs, and confirm the z.N.pkl learning "
        "curve plateaued; re-run this script and check whether ΔBIC and canonical "
        "correlation improve. If they don't, the continuity is data-driven, not "
        "an undertraining artifact.",
        "- For a discrete read if warranted: feed `cryodrgn_compare.py` for the "
        "ARI/NMI clustering view, and use the per_particle.npz JS/agreement to "
        "select a high-confidence (low-JS, agree==1) subset for refinement.",
    ]
    with open(os.path.join(args.outdir, "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n")

    print("\n".join(summary).encode("ascii", "replace").decode("ascii"))
    print(f"\n[done] wrote outputs to {args.outdir}")


if __name__ == "__main__":
    main()
