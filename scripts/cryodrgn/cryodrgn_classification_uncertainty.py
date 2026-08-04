#!/usr/bin/env python
r"""Compare cryoDRGN *classification uncertainty* ACROSS runs.

The question this answers: "how uncertain is the cryoDRGN classification, and
does that uncertainty go down when I change hyperparameters (beta, zdim, box) or
apply a focus mask?"

For every run it fits the SAME K-mode Gaussian mixture in the standardized latent
and reports two complementary, comparable numbers:

  * PER-PARTICLE uncertainty (how ambiguous is each particle's class):
        - mean max responsibility   (confidence; ->1 = certain)
        - mean normalized entropy    (uncertainty; ->0 = certain, ->1 = uniform)
        - fraction of particles with max-resp > 0.9  (confidently assigned)

  * WHETHER that confidence is EARNED (separation of the states themselves):
        - min / mean pairwise separation in pooled SD between components
        - silhouette
        - expected max-resp for 2 Gaussians min_sep apart, and the
          OVER-CONFIDENCE GAP = observed - expected.  A GMM will always look
          confident on a continuous cloud; the gap tells you whether the low
          per-particle uncertainty is real (small gap, well separated) or
          manufactured (large gap, overlapping cloud).

Using ONE fixed K across all runs in a comparison keeps the confidence numbers
apples-to-apples.  No CryoSPARC labels are used, so lower uncertainty cannot be
achieved by simply reproducing the existing classification.

Run with the cryodrgn-py310 env from the repo root, e.g. compare J1442's
hyperparameter sweep + focus-mask runs (all D=128 fullset, same N)::

    python scripts/cryodrgn/cryodrgn_classification_uncertainty.py \
      --dataset J1442 --k 3 \
      --runs results_cryodrgn/J1442_gP25_WT_POSE_BIAS/train_sweep_D128_* \
             results_cryodrgn/J1442_gP25_WT_POSE_BIAS/train_recover_D128_z16_b0p03 \
             results_cryodrgn/J1442_gP25_WT_POSE_BIAS/train_focus_z16_b0p03_s0 \
             results_cryodrgn/J1442_gP25_WT_POSE_BIAS/train_focus_z16_b0p03_s1 \
      -o results_cryodrgn/J1442_gP25_WT_POSE_BIAS/classification_uncertainty
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import pickle
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def latest_z(run_dir: str):
    best, best_ep = None, -1
    for p in glob.glob(os.path.join(run_dir, "z.*.pkl")):
        m = re.search(r"z\.(\d+)\.pkl$", p)
        if m and int(m.group(1)) > best_ep:
            best, best_ep = p, int(m.group(1))
    return best, best_ep


def load_z(path):
    with open(path, "rb") as fh:
        return np.asarray(pickle.load(fh), dtype=np.float64)


def short_label(run_dir: str):
    """Compact human label + parsed hyperparameters from the run dir name."""
    name = os.path.basename(os.path.normpath(run_dir))
    b = re.search(r"_b([0-9p.]+)", name)
    z = re.search(r"_z(\d+)", name)
    s = re.search(r"_s(\d+)", name)
    d = re.search(r"[_D](\d{2,3})", name)
    beta = float(b.group(1).replace("p", ".")) if b else np.nan
    zdim = int(z.group(1)) if z else -1
    seed = int(s.group(1)) if s else -1
    focus = "focus" in name.lower()
    tag = []
    if not np.isnan(beta):
        tag.append(f"b{beta:g}")
    if zdim > 0:
        tag.append(f"z{zdim}")
    if seed >= 0:
        tag.append(f"s{seed}")
    if focus:
        tag.insert(0, "FOCUS")
    return name, " ".join(tag) if tag else name, beta, zdim, seed, focus


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def pairwise_sep_sd(means, covs):
    """min & mean pairwise separation in pooled SD (matches overfit_check)."""
    k, d = means.shape
    seps = []
    for a in range(k):
        for b in range(a + 1, k):
            pooled = 0.5 * (np.trace(covs[a]) + np.trace(covs[b])) / d
            seps.append(np.linalg.norm(means[a] - means[b]) /
                        np.sqrt(max(pooled, 1e-12)))
    if not seps:
        return float("nan"), float("nan")
    return float(min(seps)), float(np.mean(seps))


def expected_maxresp_two_gaussians(sep_sd, n=200000, seed=0):
    """MC expected max posterior for an equal-weight 2-Gaussian mixture whose
    means are `sep_sd` pooled-SDs apart (unit variance, 1-D)."""
    if not np.isfinite(sep_sd):
        return float("nan")
    rng = np.random.default_rng(seed)
    half = n // 2
    x = np.concatenate([rng.normal(0, 1, half), rng.normal(sep_sd, 1, n - half)])
    logA = -0.5 * x ** 2
    logB = -0.5 * (x - sep_sd) ** 2
    m = np.maximum(logA, logB)
    pA, pB = np.exp(logA - m), np.exp(logB - m)
    post = np.maximum(pA, pB) / (pA + pB)
    return float(post.mean())


def score_run(Xs, k, sub, seed):
    """Fit K-mode GMM, return per-particle uncertainty + separation stats."""
    n = len(Xs)
    rng = np.random.default_rng(seed)
    idx = (rng.choice(n, sub, replace=False) if sub and n > sub else np.arange(n))
    Xfit = Xs[idx]
    gmm = GaussianMixture(k, covariance_type="full", n_init=8, max_iter=1000,
                          tol=1e-6, reg_covar=1e-6, random_state=seed).fit(Xfit)
    resp = gmm.predict_proba(Xs)                      # per-particle, full set
    max_resp = resp.max(1)
    # normalized entropy in [0,1]; 0 = certain, 1 = uniform over K
    p = np.clip(resp, 1e-12, 1.0)
    ent = -(p * np.log(p)).sum(1) / np.log(k)

    min_sep, mean_sep = pairwise_sep_sd(gmm.means_, gmm.covariances_)
    expected = expected_maxresp_two_gaussians(min_sep, seed=seed)

    try:
        hard = resp.argmax(1)[idx]
        sil = (float(silhouette_score(Xfit, hard))
               if len(set(hard)) > 1 else float("nan"))
    except Exception:
        sil = float("nan")

    return {
        "mean_max_resp": float(max_resp.mean()),
        "median_max_resp": float(np.median(max_resp)),
        "frac_conf_0.9": float((max_resp > 0.9).mean()),
        "mean_norm_entropy": float(ent.mean()),
        "median_norm_entropy": float(np.median(ent)),
        "min_sep_sd": min_sep,
        "mean_sep_sd": mean_sep,
        "silhouette": sil,
        "expected_max_resp": expected,
        "overconfidence_gap": (float(max_resp.mean() - expected)
                               if np.isfinite(expected) else float("nan")),
    }


# --------------------------------------------------------------------------- #
# Resolvable-modes (label-free K suggestion, same recipe as sweep_score)
# --------------------------------------------------------------------------- #
def resolvable_modes(Xs, kmax, sep_thresh, min_pop, seed):
    gmm = GaussianMixture(kmax, covariance_type="full", reg_covar=1e-6,
                          max_iter=500, n_init=4, random_state=seed).fit(Xs)
    hard = gmm.predict(Xs)
    pops = np.array([(hard == i).mean() for i in range(kmax)])
    k, d = gmm.means_.shape
    parent = list(range(kmax))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(kmax):
        for j in range(i + 1, kmax):
            di = gmm.means_[i] - gmm.means_[j]
            dist = np.linalg.norm(di)
            u = di / (dist + 1e-12)
            si = np.sqrt(u @ gmm.covariances_[i] @ u)
            sj = np.sqrt(u @ gmm.covariances_[j] @ u)
            if dist / (0.5 * (si + sj) + 1e-12) < sep_thresh:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(kmax):
        groups.setdefault(find(i), []).append(i)
    return int(sum(pops[m].sum() >= min_pop for m in groups.values()))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run dirs (globs ok), each with z.<N>.pkl")
    ap.add_argument("--dataset", default="dataset", help="label for plots/files")
    ap.add_argument("--k", type=int, default=0,
                    help="fixed number of classes for the comparison "
                         "(0 = auto: median label-free resolvable-modes across runs)")
    ap.add_argument("--sub", type=int, default=40000,
                    help="subsample for GMM fit / silhouette (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    run_dirs = []
    for r in args.runs:
        run_dirs.extend(sorted(glob.glob(r)) if any(c in r for c in "*?[") else [r])
    run_dirs = [r for r in run_dirs if os.path.isdir(r) and latest_z(r)[0]]
    if not run_dirs:
        raise SystemExit("no run directories with z.*.pkl matched --runs")

    # load latents
    loaded = []
    for rd in run_dirs:
        zp, ep = latest_z(rd)
        z = load_z(zp)
        Xs = StandardScaler().fit_transform(z)
        name, label, beta, zdim, seed, focus = short_label(rd)
        loaded.append(dict(dir=rd, name=name, label=label, beta=beta, zdim=zdim,
                           seed=seed, focus=focus, epoch=ep, X=Xs))

    # choose common K
    k = args.k
    if k <= 0:
        rng = np.random.default_rng(args.seed)
        modes = []
        for r in loaded:
            X = r["X"]
            idx = (rng.choice(len(X), args.sub, replace=False)
                   if args.sub and len(X) > args.sub else np.arange(len(X)))
            modes.append(resolvable_modes(X[idx], 8, 1.5, 0.04, args.seed))
        k = int(np.median(modes))
        k = max(k, 2)
        print(f"[auto] resolvable modes per run = {modes} -> common K = {k}")
    print(f"Comparing classification uncertainty at K={k} across "
          f"{len(loaded)} runs (dataset {args.dataset}).\n")

    hdr = (f"{'run':<40}{'ep':>4}{'meanMaxR':>9}{'frac>0.9':>9}"
           f"{'meanEnt':>8}{'minSep':>8}{'sil':>7}{'ocGap':>7}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for r in loaded:
        m = score_run(r["X"], k, args.sub, args.seed)
        row = dict(run=r["name"], label=r["label"], beta=r["beta"], zdim=r["zdim"],
                   seed=r["seed"], focus=int(r["focus"]), epoch=r["epoch"], k=k, **m)
        rows.append(row)
        print(f"{r['name']:<40}{r['epoch']:>4}{m['mean_max_resp']:>9.3f}"
              f"{m['frac_conf_0.9']:>9.3f}{m['mean_norm_entropy']:>8.3f}"
              f"{m['min_sep_sd']:>8.2f}{m['silhouette']:>7.3f}"
              f"{m['overconfidence_gap']:>7.3f}")

    # ---- outputs ---------------------------------------------------------- #
    csv_path = os.path.join(args.outdir, f"{args.dataset}_classification_uncertainty.csv")
    cols = ["run", "label", "beta", "zdim", "seed", "focus", "epoch", "k",
            "mean_max_resp", "median_max_resp", "frac_conf_0.9",
            "mean_norm_entropy", "median_norm_entropy",
            "min_sep_sd", "mean_sep_sd", "silhouette",
            "expected_max_resp", "overconfidence_gap"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in cols})

    with open(os.path.join(args.outdir, f"{args.dataset}_uncertainty.json"), "w") as fh:
        json.dump({"k": k, "runs": rows, "params": vars(args)}, fh,
                  indent=2, default=float)

    # ---- figure ----------------------------------------------------------- #
    order = list(range(len(rows)))
    labels = [rows[i]["label"] for i in order]
    ent = [rows[i]["mean_norm_entropy"] for i in order]
    sep = [rows[i]["min_sep_sd"] for i in order]
    obs = [rows[i]["mean_max_resp"] for i in order]
    exp = [rows[i]["expected_max_resp"] for i in order]
    colors = ["#d1495b" if rows[i]["focus"] else "#4C72B0" for i in order]

    fig, axes = plt.subplots(1, 3, figsize=(max(11, len(rows) * 1.1), 4.4))
    x = np.arange(len(order))

    axes[0].bar(x, ent, color=colors)
    axes[0].set_ylabel("mean normalized entropy  (uncertainty, lower = better)")
    axes[0].set_title(f"{args.dataset}: per-particle classification uncertainty (K={k})")

    axes[1].bar(x, sep, color=colors)
    axes[1].axhline(2.0, ls="--", c="grey", lw=1)
    axes[1].set_ylabel("min pairwise separation (pooled SD)")
    axes[1].set_title("State separation (>2 SD = genuinely distinct)")

    w = 0.4
    axes[2].bar(x - w / 2, exp, w, color="#7fb069", label="expected @ sep")
    axes[2].bar(x + w / 2, obs, w, color="#d1495b", label="observed")
    axes[2].set_ylabel("mean max responsibility")
    axes[2].set_ylim(0.5, 1.02)
    axes[2].set_title("Observed vs earned confidence (gap = over-confidence)")
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, f"{args.dataset}_uncertainty_compare.png"),
                dpi=150)
    plt.close(fig)

    print(f"\n[out] {args.outdir}/")
    print(f"      {os.path.basename(csv_path)}")
    print(f"      {args.dataset}_uncertainty_compare.png")
    print("\nRead the FOCUS run (red) against the sweep runs (blue): lower entropy "
          "AND higher separation = the focus mask genuinely reduced classification "
          "uncertainty (not just manufactured confidence).")


if __name__ == "__main__":
    main()
