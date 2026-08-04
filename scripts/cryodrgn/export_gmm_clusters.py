#!/usr/bin/env python
"""Best-fit GMM clustering of a cryoDRGN latent -> CryoSPARC .cs + RELION .star.

Fits a Gaussian Mixture in the FULL (all-zdim, standardized) latent space, picks
the number of clusters K by the BIC elbow (kneedle) with a small-population
guard, and exports one CryoSPARC-importable .cs per cluster (a row-subset of the
passthrough, so every original blob/ctf/alignments field is preserved) plus a
per-cluster index .pkl and the exact ``cryodrgn_utils write_star`` command to
make the matching RELION .star.

Latent rows map 1:1 to the passthrough (fullset run, no --ind), so cluster
indices are valid for BOTH the .cs subset and write_star --ind.

Run with the cryodrgn env from repo root::

    python scripts/cryodrgn/export_gmm_clusters.py \\
      --z results_cryodrgn/J1442_gP25_WT_POSE_BIAS/fullset_D256_z10_ep100/z.100.pkl \\
      --passthrough-cs data/gP25W6_J1442_J1497/cryosparc_P25_J1442_passthrough_particles_all_classes.cs \\
      --dataset J1442 \\
      --star-particles results_cryodrgn/J1442_gP25_WT_POSE_BIAS/inputs/particles.256.mrcs \\
      --star-ctf results_cryodrgn/J1442_gP25_WT_POSE_BIAS/inputs/ctf.pkl \\
      --star-poses results_cryodrgn/J1442_gP25_WT_POSE_BIAS/inputs/poses.pkl \\
      -o results_cryodrgn/J1442_gP25_WT_POSE_BIAS/fullset_D256_z10_ep100/gmm_clusters
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def load_z(path):
    with open(path, "rb") as fh:
        return np.asarray(pickle.load(fh), dtype=np.float64)


def save_cs(path, arr):
    """np.save via a file handle so the .cs name is kept (no .npy suffix)."""
    with open(path, "wb") as fh:
        np.save(fh, arr)


def min_sep_sd(g):
    K = g.n_components
    best = np.inf
    for i in range(K):
        for j in range(i + 1, K):
            d = g.means_[i] - g.means_[j]
            dist = np.linalg.norm(d)
            u = d / (dist + 1e-12)
            si = np.sqrt(u @ g.covariances_[i] @ u)
            sj = np.sqrt(u @ g.covariances_[j] @ u)
            best = min(best, dist / (0.5 * (si + sj) + 1e-12))
    return float(best)


def pick_elbow(ks, bics):
    x = np.asarray(ks, float); y = np.asarray(bics, float)
    xn = (x - x.min()) / (np.ptp(x) + 1e-9)
    yn = (y - y.min()) / (np.ptp(y) + 1e-9)
    p1, p2 = np.array([xn[0], yn[0]]), np.array([xn[-1], yn[-1]])
    seg = p2 - p1
    d = [abs(np.cross(seg, p1 - np.array([xn[i], yn[i]]))) / (np.linalg.norm(seg) + 1e-9)
         for i in range(len(xn))]
    return int(ks[int(np.argmax(d))])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--z", required=True)
    ap.add_argument("--passthrough-cs", required=True,
                    help="importable passthrough .cs (blob[/ctf/pose]) for ALL particles")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--k", default="auto", help="'auto' (BIC elbow) or an integer")
    ap.add_argument("--k-max", type=int, default=12)
    ap.add_argument("--min-pop", type=float, default=0.02,
                    help="reject K whose smallest cluster is below this fraction")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sub", type=int, default=30000, help="subsample for silhouette")
    ap.add_argument("--star-particles", default="<particles.mrcs>",
                    help="hudson path to the mrcs, used in the emitted write_star cmds")
    ap.add_argument("--star-ctf", default="<ctf.pkl>")
    ap.add_argument("--star-poses", default="<poses.pkl>")
    ap.add_argument("-o", "--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    z = load_z(args.z)
    n, zdim = z.shape
    print(f"[load] latent {z.shape} from {args.z}")
    Xs = StandardScaler().fit_transform(z)

    rng = np.random.default_rng(args.seed)
    sidx = (rng.choice(n, args.sub, replace=False)
            if args.sub and n > args.sub else np.arange(n))

    # ---- model selection ------------------------------------------------- #
    ks = list(range(2, args.k_max + 1))
    rows = []
    print(f"\n{'K':>3}{'BIC':>14}{'minSep':>8}{'silhou':>8}{'minPop%':>8}")
    for k in ks:
        g = GaussianMixture(k, covariance_type="full", reg_covar=1e-6,
                            n_init=4, max_iter=500, random_state=args.seed).fit(Xs)
        hard = g.predict(Xs)
        pops = np.array([(hard == c).mean() for c in range(k)])
        try:
            sil = silhouette_score(Xs[sidx], g.predict(Xs[sidx]))
        except Exception:
            sil = float("nan")
        bic = g.bic(Xs)
        rows.append(dict(k=k, bic=bic, minsep=min_sep_sd(g), sil=sil, minpop=pops.min()))
        print(f"{k:>3}{bic:>14.0f}{rows[-1]['minsep']:>8.2f}{sil:>8.3f}{pops.min()*100:>8.2f}")

    if args.k != "auto":
        bestK = int(args.k)
    else:
        elbow = pick_elbow([r["k"] for r in rows], [r["bic"] for r in rows])
        # guard: don't pick a K whose smallest cluster is below --min-pop
        ok = [r["k"] for r in rows if r["minpop"] >= args.min_pop]
        bestK = elbow if elbow in ok else (max(ok) if ok else rows[0]["k"])
        print(f"\n[select] BIC elbow K={elbow}; min-pop guard -> best K = {bestK}")
    print(f"[select] using K = {bestK}")

    # ---- final fit, PC1-ordered labels ----------------------------------- #
    gmm = GaussianMixture(bestK, covariance_type="full", reg_covar=1e-6,
                          n_init=10, max_iter=500, random_state=args.seed).fit(Xs)
    resp = gmm.predict_proba(Xs)
    hard = resp.argmax(1)
    pca = PCA(2, random_state=args.seed).fit(Xs)
    order = np.argsort(pca.transform(gmm.means_)[:, 0])
    relabel = np.empty(bestK, int); relabel[order] = np.arange(bestK)
    hard = relabel[hard]
    resp = resp[:, order]
    names = [f"c{i}" for i in range(bestK)]

    # ---- load passthrough, export per-cluster .cs + ind.pkl --------------- #
    cs = np.load(args.passthrough_cs)
    if len(cs) != n:
        raise SystemExit(f"passthrough has {len(cs)} rows but latent has {n}; "
                         "wrong passthrough or this was an --ind run.")
    uids = cs["uid"]

    star_lines = ["#!/usr/bin/env bash", "set -euo pipefail",
                  'DIR="$(cd "$(dirname "$0")" && pwd)"',
                  "# Run on hudson (cryodrgn env) from anywhere; finds its own ind_*.pkl.",
                  "# The mrcs/ctf/poses below should be ABSOLUTE hudson paths (or repo-relative",
                  "# if you run from the repo root)."]
    print(f"\n{'cluster':<8}{'N':>10}{'frac%':>8}")
    for i in range(bestK):
        sel = hard == i
        rows_i = np.where(sel)[0]
        save_cs(os.path.join(args.outdir, f"{args.dataset}_gmm_k{bestK}_{names[i]}.cs"),
                cs[rows_i])
        indp = os.path.join(args.outdir, f"ind_{names[i]}.pkl")
        with open(indp, "wb") as fh:
            pickle.dump(rows_i.astype(np.int64), fh)
        print(f"{names[i]:<8}{len(rows_i):>10,}{sel.mean()*100:>8.1f}")
        star_lines.append(
            f'cryodrgn_utils write_star "{args.star_particles}" '
            f'-o "$DIR/{args.dataset}_gmm_k{bestK}_{names[i]}.star" '
            f'--ctf "{args.star_ctf}" --poses "{args.star_poses}" '
            f'--ind "$DIR/ind_{names[i]}.pkl"')

    # sidecar assignments + star script
    with open(os.path.join(args.outdir, f"{args.dataset}_gmm_assignments.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["uid", "cluster", "max_resp"] + [f"resp_{c}" for c in names])
        for r in range(n):
            w.writerow([int(uids[r]), names[hard[r]], f"{resp[r].max():.4f}"]
                       + [f"{resp[r, c]:.4f}" for c in range(bestK)])
    star_sh = os.path.join(args.outdir, "write_star.sh")
    with open(star_sh, "w", newline="\n") as fh:
        fh.write("\n".join(star_lines) + "\n")

    print(f"\n[out] {args.outdir}/")
    print(f"      {bestK} .cs (CryoSPARC import) + ind_c*.pkl + "
          f"{args.dataset}_gmm_assignments.csv + write_star.sh")
    print("[star] make .star: run write_star.sh on hudson (or scp the ind_c*.pkl up), "
          "e.g.\n       cryodrgn_utils write_star <mrcs> -o cN.star --ctf ctf.pkl "
          "--poses poses.pkl --ind ind_cN.pkl")


if __name__ == "__main__":
    main()
