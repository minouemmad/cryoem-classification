#!/usr/bin/env python
"""Carve a cryoDRGN latent into per-blob --ind files for divide-and-conquer.

The global J1442 latent is robustly ~3 blobs; the "missing" states (P9, P10)
are subtle sub-states INSIDE those blobs.  To look for them, retrain cryoDRGN on
ONE blob at a time -- with the competing global variance removed, a within-blob
sub-state has room to separate.

This clusters the full standardized latent into K modes (default 3, matching the
resolvable-mode count), and writes one cryoDRGN ``--ind`` pickle per mode
(indices into the ORIGINAL particle stack -- valid because a fullset z row i
corresponds to particle i).  If --cs is given, each mode is NAMED by its majority
CryoSPARC class so you know which ind file is the P8 blob -- labels are used ONLY
to name the blob, NOT to define the sub-states (the sub-state search stays
unsupervised: it happens when you retrain on the blob).

Run with the cryodrgn-py310 env from repo root::

    python scripts/cryodrgn/make_subset_ind.py \\
      --z results_cryodrgn/J1442_gP25_WT_POSE_BIAS/train_recover_D128_z16_b0p03/z.50.pkl \\
      --passthrough-cs data/cryosparc_P25_J1442_passthrough_particles_all_classes.cs \\
      --cs data/cryosparc_P25_J1442_00000_particles.cs --n-dummies 6 \\
      -k 3 -o results_cryodrgn/J1442_gP25_WT_POSE_BIAS/subset_inds

Then retrain on the P8 blob (train-only, then score for a split)::

    IND=results_cryodrgn/J1442_gP25_WT_POSE_BIAS/subset_inds/ind_c2_P8.pkl \\
    ZDIM=16 BETA=0.03 SKIP_ANALYZE=1 SKIP_EXPORT=1 \\
      bash scripts/cryodrgn/j1442_recover_states.sh
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
_REPO = os.path.dirname(_SCRIPTS)
for _p in (_REPO, _SCRIPTS, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import cryodrgn_latent_gmm as clg


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--z", required=True, help="cryoDRGN z.<N>.pkl (fullset run)")
    ap.add_argument("--passthrough-cs", required=True,
                    help="passthrough .cs (uid order == z row order)")
    ap.add_argument("--cs", default=None,
                    help="CryoSPARC particles .cs with class posteriors (to NAME blobs)")
    ap.add_argument("--protein-idx", type=int, nargs="+", default=None,
                    help="protein class indices (e.g. 6 7 8 for J1442)")
    ap.add_argument("--n-dummies", type=int, default=6)
    ap.add_argument("-k", type=int, default=3, help="number of blobs to carve")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    z = clg.load_latent(args.z)
    n = len(z)
    print(f"[load] latent {z.shape} from {args.z}")
    Xs = StandardScaler().fit_transform(z)

    gmm = GaussianMixture(args.k, covariance_type="full", reg_covar=1e-6,
                          max_iter=500, n_init=10, random_state=args.seed).fit(Xs)
    hard = gmm.predict(Xs)

    # stable ordering: label blobs c0..c{k-1} left->right along PC1
    pca = PCA(n_components=2, random_state=args.seed).fit(Xs)
    order = np.argsort(pca.transform(gmm.means_)[:, 0])
    relabel = np.empty(args.k, dtype=int)
    relabel[order] = np.arange(args.k)
    hard = relabel[hard]

    # optional CryoSPARC naming (labels only NAME the blob)
    name_of = {i: f"c{i}" for i in range(args.k)}
    if args.cs and args.protein_idx:
        z_a, _, cryo_hard_a, _, _ = clg.align_z_to_posteriors(
            z, args.passthrough_cs, args.cs, args.n_dummies, args.protein_idx)
        # align returns matched rows in z order; for a fullset run this is all rows
        if len(cryo_hard_a) == n:
            for i in range(args.k):
                sel = hard == i
                maj = np.bincount(cryo_hard_a[sel],
                                  minlength=int(cryo_hard_a.max()) + 1).argmax()
                frac = float((cryo_hard_a[sel] == maj).mean())
                pidx = args.protein_idx[maj] if maj < len(args.protein_idx) else maj
                name_of[i] = f"c{i}_P{pidx}"
                print(f"[name] blob c{i}: {sel.sum():,} particles, "
                      f"majority CryoSPARC P{pidx} ({frac*100:.0f}%)")
        else:
            print(f"[name] WARNING matched {len(cryo_hard_a)} != {n} rows; "
                  "skipping names (partial uid overlap)")

    print(f"\n{'blob':<12}{'particles':>12}{'ind file':>34}")
    print("-" * 58)
    for i in range(args.k):
        ind = np.where(hard == i)[0].astype(np.int64)
        fn = f"ind_{name_of[i]}.pkl"
        with open(os.path.join(args.outdir, fn), "wb") as fh:
            pickle.dump(ind, fh)
        print(f"{name_of[i]:<12}{len(ind):>12,}{fn:>34}")

    print(f"\n[out] {args.outdir}/  ({args.k} ind_*.pkl files)")
    print("Retrain on a blob:  IND=<outdir>/ind_c<i>_P<n>.pkl SKIP_ANALYZE=1 "
          "SKIP_EXPORT=1 bash scripts/cryodrgn/j1442_recover_states.sh")
    print("Then score for a split:  python scripts/cryodrgn/cryodrgn_sweep_score.py "
          "--runs <that run dir> -o <...>/subset_score")


if __name__ == "__main__":
    main()
