#!/usr/bin/env python
r"""Export cryoDRGN latent GMM classes to CryoSPARC .cs and RELION .star.

Fits a Gaussian Mixture Model in the full standardized cryoDRGN latent space,
selects K by a component-separability sweep (largest K whose minimum pairwise
component separation is >= --sep-thresh SD), and exports one particle subset per
GMM component.  Each subset is written as a CryoSPARC .cs (a row subset of the
self-contained passthrough .cs, i.e. carrying blob/CTF/pose) and converted to a
RELION 3.1 .star with pyem's csparc2star.

Latent rows are matched to the passthrough .cs by ROW ORDER, which is valid for
full-set cryoDRGN runs (the training stack is built in passthrough order).  The
script asserts that len(z) == len(passthrough).

Run from the repo root with the cryodrgn-py310 environment, e.g.::

    cryodrgn-py310\Scripts\python.exe scripts\cryodrgn\export_gmm_classes.py ^
        --z results_cryodrgn\...\z.100.pkl ^
        --passthrough-cs data\...\..._blob.cs ^
        --dataset J2708_D256ep100 -o results_cryodrgn\...\gmm_export
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import subprocess
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

DEFAULT_CSPARC2STAR = str(
    Path(__file__).resolve().parents[2] / "cryodrgn-py310" / "Scripts" / "csparc2star.exe"
)


def load_latent(path: str) -> np.ndarray:
    with open(path, "rb") as fh:
        z = pickle.load(fh)
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z[:, None]
    return z


def min_separation_sd(gmm: GaussianMixture) -> tuple:
    """Minimum and mean pairwise component separation in SD units."""
    K = gmm.n_components
    means = gmm.means_
    seps = []
    for i in range(K):
        for j in range(i + 1, K):
            d = means[i] - means[j]
            dist = np.linalg.norm(d)
            u = d / (dist + 1e-12)
            si = np.sqrt(u @ gmm.covariances_[i] @ u)
            sj = np.sqrt(u @ gmm.covariances_[j] @ u)
            seps.append(dist / (0.5 * (si + sj) + 1e-12))
    if not seps:
        return float("inf"), float("inf")
    return float(np.min(seps)), float(np.mean(seps))


def separation_sweep(Xs: np.ndarray, k_min: int, k_max: int, seed: int) -> list:
    rows = []
    for k in range(k_min, k_max + 1):
        gmm = GaussianMixture(
            n_components=k, covariance_type="full", reg_covar=1e-6,
            max_iter=500, n_init=5, random_state=seed,
        ).fit(Xs)
        min_sep, mean_sep = min_separation_sd(gmm)
        rows.append({
            "k": k, "bic": float(gmm.bic(Xs)),
            "converged": bool(gmm.converged_),
            "min_weight": float(gmm.weights_.min()),
            "min_sep_sd": min_sep, "mean_sep_sd": mean_sep,
        })
        print(f"  K={k:2d}  BIC={rows[-1]['bic']:.1f}  min_sep={min_sep:.2f} SD  "
              f"mean_sep={mean_sep:.2f} SD  converged={rows[-1]['converged']}  "
              f"min_w={rows[-1]['min_weight']:.4f}")
    return rows


def select_k_by_separation(rows: list, sep_thresh: float) -> int:
    eligible = [r for r in rows if r["min_sep_sd"] >= sep_thresh]
    if eligible:
        best = max(eligible, key=lambda r: r["k"])["k"]
        print(f"      -> selected K={best} (largest K with min_sep >= {sep_thresh} SD)")
    else:
        best = max(rows, key=lambda r: r["min_sep_sd"])["k"]
        print(f"      -> WARNING: no K reached {sep_thresh} SD; "
              f"falling back to most-separable K={best}")
    return best


def plot_selection_sweep(rows, best_k, sep_thresh, dataset, out_path):
    ks = [r["k"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(ks, [r["min_sep_sd"] for r in rows], "o-", color="#d62728")
    ax1.axhline(sep_thresh, color="black", ls="--",
                label=f"separability threshold = {sep_thresh} SD")
    ax1.axvline(best_k, color="crimson", ls=":", label=f"chosen K = {best_k}")
    ax1.set_xlabel("number of GMM components K")
    ax1.set_ylabel("min pairwise component separation (SD)")
    ax1.set_title("Component separability")
    ax1.legend()
    ax2.plot(ks, [r["bic"] for r in rows], "o-", color="#1f77b4")
    ax2.axvline(best_k, color="crimson", ls=":", label=f"chosen K = {best_k}")
    ax2.set_xlabel("number of GMM components K")
    ax2.set_ylabel("BIC (lower = better)")
    ax2.set_title("BIC for reference")
    ax2.legend()
    fig.suptitle(f"{dataset} latent-space GMM model selection")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def convert_cs_to_star(csparc2star: str, cs_path: Path, star_path: Path) -> str:
    """Convert a self-contained CryoSPARC .cs to a RELION .star via csparc2star."""
    candidates = [csparc2star]
    if csparc2star != "csparc2star":
        candidates.append("csparc2star")  # fall back to PATH lookup
    errors = []
    for exe in candidates:
        cmd = [exe, str(cs_path), str(star_path)]
        try:
            proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if proc.stdout.strip():
                print(f"          converter: {proc.stdout.strip().splitlines()[-1]}")
            return " ".join(cmd)
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            msg = str(exc)
            if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
                msg = f"{msg}; stderr={exc.stderr.strip()}"
            errors.append(f"{' '.join(cmd)} -> {msg}")
    raise RuntimeError("Could not convert .cs to .star:\n" + "\n".join(errors))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--z", required=True, help="cryoDRGN latent file (z.*.pkl)")
    ap.add_argument("--passthrough-cs", required=True,
                    help="self-contained passthrough .cs (blob/CTF/pose)")
    ap.add_argument("--dataset", required=True, help="label used for filenames/titles")
    ap.add_argument("--k", default="auto",
                    help="'auto' (separability sweep) or a fixed integer K")
    ap.add_argument("--select-k", default="2,12",
                    help="K sweep range MIN,MAX for the selection plot (default 2,12)")
    ap.add_argument("--sep-thresh", type=float, default=2.0,
                    help="min pairwise component separation (SD) to accept a K")
    ap.add_argument("--min-resp", type=float, default=0.0,
                    help="optional confident-subset threshold on max responsibility")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csparc2star", default=DEFAULT_CSPARC2STAR,
                    help="path to csparc2star executable")
    ap.add_argument("-o", "--outdir", required=True)
    args = ap.parse_args()

    k_min, k_max = (int(x) for x in args.select_k.split(","))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading latent from {args.z}")
    z = load_latent(args.z)
    n, zdim = z.shape
    print(f"      latent shape: {z.shape}")

    print(f"[2/5] Loading passthrough .cs {args.passthrough_cs}")
    passthrough = np.load(args.passthrough_cs)
    if "uid" not in passthrough.dtype.names:
        raise SystemExit("ERROR: passthrough .cs has no uid field")
    if len(passthrough) != n:
        raise SystemExit(
            f"ERROR: length mismatch z={n} vs passthrough={len(passthrough)}; "
            "row-order matching requires equal lengths (full-set run).")
    print(f"      passthrough particles: {len(passthrough):,} (row-order match OK)")

    print("[3/5] Standardizing latent and running separability sweep")
    scaler = StandardScaler().fit(z)
    Xs = scaler.transform(z)
    rows = separation_sweep(Xs, k_min, k_max, args.seed)
    auto_k = select_k_by_separation(rows, args.sep_thresh)
    if str(args.k).lower() == "auto":
        best_k = auto_k
    else:
        best_k = int(args.k)
        print(f"      -> using user-specified K={best_k} (auto sweep suggested {auto_k})")
    plot_selection_sweep(rows, best_k, args.sep_thresh, args.dataset,
                         outdir / f"{args.dataset}_gmm_selection_sweep.png")

    print(f"[4/5] Fitting final GMM with K={best_k}")
    gmm = GaussianMixture(
        n_components=best_k, covariance_type="full", reg_covar=1e-6,
        max_iter=500, n_init=10, random_state=args.seed,
    ).fit(Xs)
    resp = gmm.predict_proba(Xs)
    hard = resp.argmax(axis=1)
    maxresp = resp.max(axis=1)
    min_sep, mean_sep = min_separation_sd(gmm)
    print(f"      min component separation: {min_sep:.2f} SD (mean {mean_sep:.2f} SD)")

    # Order components left-to-right along PC1 for stable, interpretable labels.
    pca = PCA(n_components=min(3, zdim), random_state=args.seed).fit(Xs)
    order = np.argsort(pca.transform(gmm.means_)[:, 0])
    relabel = np.empty(best_k, dtype=int)
    relabel[order] = np.arange(best_k)
    hard_ordered = relabel[hard]
    resp_ordered = resp[:, order]
    names = [f"c{i}" for i in range(best_k)]
    print("      population fractions: " +
          "  ".join(f"{names[i]}={(hard_ordered == i).mean()*100:.1f}%"
                    for i in range(best_k)))

    # Sidecar CSV with per-particle assignment and responsibilities.
    with open(outdir / f"{args.dataset}_gmm_assignments.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["uid", "cluster", "max_resp"] + [f"resp_{names[i]}" for i in range(best_k)])
        uids = passthrough["uid"]
        for i in range(n):
            w.writerow([int(uids[i]), names[hard_ordered[i]], f"{maxresp[i]:.6f}"]
                       + [f"{resp_ordered[i, j]:.6f}" for j in range(best_k)])
    print(f"      wrote {args.dataset}_gmm_assignments.csv")

    print("[5/5] Exporting per-cluster .cs and .star")
    summary = []
    for k in range(best_k):
        mask = hard_ordered == k
        if args.min_resp > 0:
            mask = mask & (maxresp >= args.min_resp)
        n_k = int(mask.sum())
        if n_k == 0:
            print(f"      {names[k]}: empty, skipping")
            continue
        cs_name = outdir / f"{args.dataset}_gmm_k{best_k}_{names[k]}.cs"
        star_name = outdir / f"{args.dataset}_gmm_k{best_k}_{names[k]}.star"
        with open(cs_name, "wb") as fh:
            np.save(fh, passthrough[mask])
        try:
            cmd = convert_cs_to_star(args.csparc2star, cs_name, star_name)
            star_ok, star_msg = True, f"via {cmd}"
        except RuntimeError as exc:
            star_ok, star_msg = False, f"FAILED: {exc}"
        print(f"      {names[k]}: {n_k:,} particles -> {cs_name.name}, "
              f"{star_name.name} ({'OK' if star_ok else 'FAIL'})")
        summary.append({
            "cluster": names[k], "n_particles": n_k, "fraction": n_k / n,
            "mean_max_resp": float(maxresp[mask].mean()),
            "cs": str(cs_name), "star": str(star_name), "star_ok": star_ok,
        })

    import pandas as pd
    pd.DataFrame(summary).to_csv(outdir / f"{args.dataset}_gmm_subset_summary.csv", index=False)
    print(f"\nDone (K={best_k}, min_sep={min_sep:.2f} SD). Outputs in {outdir}")
    for s in summary:
        print(f"  {s['cluster']}: {s['n_particles']:,} ({s['fraction']*100:.1f}%) "
              f"star={'OK' if s['star_ok'] else 'FAIL'}")


if __name__ == "__main__":
    main()
