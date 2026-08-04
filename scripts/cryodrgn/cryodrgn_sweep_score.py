#!/usr/bin/env python
"""Rank cryoDRGN sweep runs by a NON-CIRCULAR "recoverable states" objective.

The problem with tuning cryoDRGN for "more intermediate states" is choosing a
score that does NOT just reward over-splitting and does NOT lean on the very
CryoSPARC labels we suspect are incomplete.  This scorer uses two label-free
signals:

  1. RESOLVABLE MODES (per run): fit a generous K-component GMM in the FULL
     standardized latent, then MERGE components whose separation is below
     `--sep-thresh` SD (single-linkage on pairwise mean-separation).  The number
     of surviving modes = how many genuinely separated blobs the latent holds.
     Merging is what stops us from counting slices of one cloud as "states".

  2. REPRODUCIBILITY (per config, needs >=2 seeds): canonical correlation between
     the latents of two seeds trained on the SAME particles.  A mode that only
     shows up at one seed is not real.  High canonical correlation => the
     coordinate (and therefore its modes) is data-driven, not a seed artifact.

Neither uses class labels, so maximizing the score cannot be satisfied by simply
reproducing the existing CryoSPARC classification.  A CryoSPARC overlay is
printed ONLY for interpretation if --cs is given; it never enters the ranking.

Run with the cryodrgn-py310 env from repo root::

    python scripts/cryodrgn/cryodrgn_sweep_score.py \\
      --runs results_cryodrgn/J1442_gP25_WT_POSE_BIAS/train_sweep_* \\
      -o results_cryodrgn/J1442_gP25_WT_POSE_BIAS/sweep_leaderboard
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (_REPO, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

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
    """Return (path, epoch) of the highest-numbered z.<N>.pkl in run_dir."""
    best, best_ep = None, -1
    for p in glob.glob(os.path.join(run_dir, "z.*.pkl")):
        m = re.search(r"z\.(\d+)\.pkl$", p)
        if m and int(m.group(1)) > best_ep:
            best, best_ep = p, int(m.group(1))
    return best, best_ep


def load_z(path):
    with open(path, "rb") as fh:
        return np.asarray(pickle.load(fh), dtype=np.float64)


def parse_tag(run_dir: str):
    """Pull beta/zdim/seed out of a train_sweep_/train_recover_ dir name."""
    name = os.path.basename(os.path.normpath(run_dir))
    z = re.search(r"_z(\d+)", name)
    b = re.search(r"_b([0-9p.]+)", name)
    s = re.search(r"_s(\d+)", name)
    beta = float(b.group(1).replace("p", ".")) if b else np.nan
    zdim = int(z.group(1)) if z else -1
    seed = int(s.group(1)) if s else -1
    return name, beta, zdim, seed


# --------------------------------------------------------------------------- #
# Objective pieces (label-free)
# --------------------------------------------------------------------------- #
def pairwise_sep_sd(gmm):
    """(K,K) matrix of mean separation in SD along each connecting axis."""
    K = gmm.n_components
    M = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            d = gmm.means_[i] - gmm.means_[j]
            dist = np.linalg.norm(d)
            u = d / (dist + 1e-12)
            si = np.sqrt(u @ gmm.covariances_[i] @ u)
            sj = np.sqrt(u @ gmm.covariances_[j] @ u)
            M[i, j] = M[j, i] = dist / (0.5 * (si + sj) + 1e-12)
    return M


def resolvable_modes(Xs, kmax, sep_thresh, min_pop, seed):
    """Fit K=kmax GMM, single-linkage merge comps with sep < sep_thresh, then
    drop merged groups whose total population < min_pop.  Returns
    (n_modes, min_sep_full, mode_pops)."""
    gmm = GaussianMixture(kmax, covariance_type="full", reg_covar=1e-6,
                          max_iter=500, n_init=6, random_state=seed).fit(Xs)
    hard = gmm.predict(Xs)
    pops = np.array([(hard == i).mean() for i in range(kmax)])
    sep = pairwise_sep_sd(gmm)

    # single-linkage merge: union components joined by an edge sep < sep_thresh
    parent = list(range(kmax))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(kmax):
        for j in range(i + 1, kmax):
            if sep[i, j] < sep_thresh:
                parent[find(i)] = find(j)

    groups = {}
    for i in range(kmax):
        groups.setdefault(find(i), []).append(i)
    group_pops = [float(pops[members].sum()) for members in groups.values()]
    n_modes = int(sum(p >= min_pop for p in group_pops))

    iu = np.triu_indices(kmax, 1)
    min_sep_full = float(sep[iu].min()) if len(iu[0]) else float("nan")
    return n_modes, min_sep_full, sorted(group_pops, reverse=True)


def canonical_correlations(A, B):
    """Canonical correlations between two matched (n x d) latents, descending."""
    A = A - A.mean(0)
    B = B - B.mean(0)
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.clip(s, 0, 1)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run directories (globs ok), each with z.<N>.pkl")
    ap.add_argument("--kmax", type=int, default=8,
                    help="components to fit before merging (default 8)")
    ap.add_argument("--sep-thresh", type=float, default=1.5,
                    help="merge components closer than this many SD (default 1.5; "
                         ">2 = strict/only very distinct survive)")
    ap.add_argument("--min-pop", type=float, default=0.04,
                    help="drop merged modes below this population fraction")
    ap.add_argument("--sub", type=int, default=40000,
                    help="subsample for silhouette/GMM speed (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    run_dirs = []
    for r in args.runs:
        run_dirs.extend(sorted(glob.glob(r)) if any(c in r for c in "*?[") else [r])
    run_dirs = [r for r in run_dirs if os.path.isdir(r)]
    if not run_dirs:
        raise SystemExit("no run directories matched --runs")

    rng = np.random.default_rng(args.seed)
    rows = []
    latents = {}  # name -> standardized latent (for reproducibility)
    print(f"{'run':<38}{'beta':>6}{'zdim':>5}{'seed':>5}{'ep':>4}"
          f"{'modes':>7}{'minSep':>8}{'sil':>7}{'PC1%':>6}{'PC2%':>6}")
    print("-" * 100)
    for rd in run_dirs:
        zp, ep = latest_z(rd)
        if zp is None:
            print(f"{os.path.basename(rd):<38}  (no z.*.pkl)")
            continue
        name, beta, zdim, seed = parse_tag(rd)
        z = load_z(zp)
        Xs = StandardScaler().fit_transform(z)
        latents[name] = (Xs, beta, zdim, seed)

        # PCA variance (report only)
        u, sv, _ = np.linalg.svd(Xs - Xs.mean(0), full_matrices=False)
        var = (sv ** 2) / (sv ** 2).sum()
        pc1, pc2 = float(var[0]), float(var[1] if len(var) > 1 else 0)

        idx = (rng.choice(len(Xs), args.sub, replace=False)
               if args.sub and len(Xs) > args.sub else np.arange(len(Xs)))
        Xsub = Xs[idx]
        n_modes, min_sep, mode_pops = resolvable_modes(
            Xsub, args.kmax, args.sep_thresh, args.min_pop, args.seed)
        try:
            km = GaussianMixture(max(n_modes, 2), covariance_type="full",
                                 random_state=args.seed).fit(Xsub).predict(Xsub)
            sil = float(silhouette_score(Xsub, km)) if len(set(km)) > 1 else float("nan")
        except Exception:
            sil = float("nan")

        rows.append(dict(run=name, beta=beta, zdim=zdim, seed=seed, epoch=ep,
                         n_modes=n_modes, min_sep=min_sep, silhouette=sil,
                         pc1=pc1, pc2=pc2, mode_pops=mode_pops))
        print(f"{name:<38}{beta:>6.3f}{zdim:>5}{seed:>5}{ep:>4}"
              f"{n_modes:>7}{min_sep:>8.2f}{sil:>7.3f}{pc1*100:>6.1f}{pc2*100:>6.1f}")

    # ----- reproducibility per (beta, zdim): canonical corr across seeds ------ #
    print("\nReproducibility (canonical correlation between seeds, same config):")
    repro = {}
    from collections import defaultdict
    by_cfg = defaultdict(list)
    for name, (Xs, beta, zdim, seed) in latents.items():
        by_cfg[(beta, zdim)].append((seed, name, Xs))
    for (beta, zdim), lst in sorted(by_cfg.items()):
        if len(lst) < 2:
            continue
        lst.sort()
        (s0, n0, X0), (s1, n1, X1) = lst[0], lst[1]
        if X0.shape[0] != X1.shape[0]:
            print(f"  b={beta} z={zdim}: seed latents differ in N; skip")
            continue
        cc = canonical_correlations(X0, X1)
        mean_top = float(np.mean(cc[:min(len(cc), zdim)]))
        repro[(beta, zdim)] = mean_top
        print(f"  b={beta:<5} z={zdim:<3} seeds {s0}/{s1}: "
              f"mean canon corr {mean_top:.3f}  (top5 "
              f"{np.array2string(cc[:5], precision=2, floatmode='fixed')})")

    # ----- leaderboard: reproducible modes first ------------------------------ #
    for r in rows:
        r["repro"] = repro.get((r["beta"], r["zdim"]), float("nan"))
    # config-level: average modes across seeds, gate by reproducibility
    cfg_rows = defaultdict(list)
    for r in rows:
        cfg_rows[(r["beta"], r["zdim"])].append(r)
    board = []
    for (beta, zdim), rs in cfg_rows.items():
        modes = [r["n_modes"] for r in rs]
        board.append(dict(
            beta=beta, zdim=zdim, n_seeds=len(rs),
            modes_mean=float(np.mean(modes)), modes_min=int(np.min(modes)),
            modes_agree=int(len(set(modes)) == 1),
            repro=repro.get((beta, zdim), float("nan")),
            min_sep_mean=float(np.mean([r["min_sep"] for r in rs])),
            pc1_mean=float(np.mean([r["pc1"] for r in rs]))))
    # rank: reproducible AND consistent modes win; then more modes; then separation
    board.sort(key=lambda d: (
        -(d["repro"] if d["repro"] == d["repro"] else -1),
        -d["modes_min"], -d["min_sep_mean"]))

    print("\n================= LEADERBOARD (config level) =================")
    print(f"{'beta':>6}{'zdim':>5}{'seeds':>6}{'modes(min/mean)':>17}"
          f"{'agree':>6}{'repro':>7}{'minSep':>8}")
    for d in board:
        rp = f"{d['repro']:.3f}" if d["repro"] == d["repro"] else "  -  "
        print(f"{d['beta']:>6.3f}{d['zdim']:>5}{d['n_seeds']:>6}"
              f"{d['modes_min']:>8}/{d['modes_mean']:>7.1f}{d['modes_agree']:>6}"
              f"{rp:>7}{d['min_sep_mean']:>8.2f}")

    with open(os.path.join(args.outdir, "sweep_scores.json"), "w") as fh:
        json.dump({"runs": rows, "leaderboard": board,
                   "params": vars(args)}, fh, indent=2, default=float)

    # CSV
    import csv
    with open(os.path.join(args.outdir, "sweep_runs.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "beta", "zdim", "seed", "epoch", "n_modes",
                    "min_sep", "silhouette", "pc1", "pc2", "repro"])
        for r in rows:
            w.writerow([r["run"], r["beta"], r["zdim"], r["seed"], r["epoch"],
                        r["n_modes"], f"{r['min_sep']:.3f}", f"{r['silhouette']:.3f}",
                        f"{r['pc1']:.3f}", f"{r['pc2']:.3f}",
                        f"{r['repro']:.3f}" if r["repro"] == r["repro"] else ""])

    # bar chart of modes per run
    if rows:
        fig, ax = plt.subplots(figsize=(max(6, len(rows) * 0.6), 4))
        labels = [f"b{r['beta']}\nz{r['zdim']}s{r['seed']}" for r in rows]
        ax.bar(range(len(rows)), [r["n_modes"] for r in rows],
               color="#4C72B0")
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel(f"resolvable modes (sep>{args.sep_thresh} SD)")
        ax.set_title("cryoDRGN sweep: label-free recoverable-state count")
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "sweep_modes.png"), dpi=150)
        plt.close(fig)

    print(f"\n[out] {args.outdir}/  (sweep_scores.json, sweep_runs.csv, sweep_modes.png)")
    print("Pick the config with the MOST modes that are also REPRODUCIBLE "
          "(high canon corr), then run the full over-cluster export + CryoSPARC "
          "ab-initio on that one only.")


if __name__ == "__main__":
    main()
