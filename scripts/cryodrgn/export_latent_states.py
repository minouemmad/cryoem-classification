#!/usr/bin/env python
"""Export cryoDRGN latent 'states' as CryoSPARC-importable particle .cs sets.

For J264 the free-energy landscape is a single continuous basin, so there are no
discrete energetic states to export. The honest, model-free thing to hand to
CryoSPARC is therefore a set of **bins along a principal coordinate** (default
PC1, the dominant conformational axis): refine each bin and you get the motion as
a series of maps, not an over-claimed set of 'classes'. (k-means/GMM are avoided
here because they impose a fixed K and manufacture boundaries on a continuum.)

Each output is a slice of the passthrough (blob/pose intact) -> CryoSPARC Import
Particle Stack -> NU-refine. An assignments CSV (uid -> bin) is also written so
the same split can instead be applied by uid inside CryoSPARC (e.g. if CTF must
come from the parent job).

Run (reusing the ind-filtered passthrough the landscape run already built)::

    ./cryodrgn-py310/Scripts/python.exe scripts/cryodrgn/export_latent_states.py \
      --z results_cryodrgn/J264_real/train_final_retry/z.40.pkl \
      --passthrough results_cryodrgn/conformational_landscape/J264_9class_D256_ep40/_passthrough_kept.npy \
      --pcs 1 --bins 5 \
      -o results_cryodrgn/latent_state_exports/J264_ep40
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for p in (_REPO, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import cryodrgn_latent_gmm as clg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--z", required=True)
    ap.add_argument("--passthrough", required=True,
                    help="passthrough .cs (or ind-filtered .npy) aligned to z by row")
    ap.add_argument("--ind", default="", help="ind_keep.pkl if passthrough is the full stack")
    ap.add_argument("--pcs", default="1", help="comma list of PCs to bin, e.g. '1,2,3'")
    ap.add_argument("--bins", type=int, default=5, help="quantile bins per PC")
    ap.add_argument("-o", "--outdir", required=True)
    args = ap.parse_args()

    z = clg.load_latent(args.z)
    pt = np.load(args.passthrough)
    if args.ind:
        idx = np.sort(np.asarray(pickle.load(open(args.ind, "rb"))).ravel().astype(np.int64))
        pt = pt[idx]
    if len(pt) != len(z):
        m = min(len(pt), len(z))
        print(f"[warn] passthrough {len(pt)} != z {len(z)}; using leading {m}")
        pt, z = pt[:m], z[:m]
    uid = pt["uid"].astype(np.uint64) if "uid" in (pt.dtype.names or ()) else np.arange(len(pt))

    zs = StandardScaler().fit_transform(z)
    pcs = PCA(3).fit_transform(zs)
    os.makedirs(args.outdir, exist_ok=True)

    for pc in [int(x) for x in args.pcs.split(",")]:
        x = pcs[:, pc - 1]
        edges = np.quantile(x, np.linspace(0, 1, args.bins + 1))
        edges[0] -= 1e-9; edges[-1] += 1e-9
        lab = np.digitize(x, edges[1:-1])
        sub = os.path.join(args.outdir, f"pc{pc}_bins")
        os.makedirs(sub, exist_ok=True)
        with open(os.path.join(sub, f"pc{pc}_bin_assignments.csv"), "w") as f:
            f.write("uid,bin,pc_value\n")
            for u, b, v in zip(uid.tolist(), lab.tolist(), x.tolist()):
                f.write(f"{int(u)},{b + 1},{v:.4f}\n")
        for b in range(args.bins):
            sel = lab == b
            subset = pt[sel]
            out = os.path.join(sub, f"pc{pc}_bin{b + 1}_particles.cs")
            with open(out, "wb") as fh:
                np.save(fh, subset)
            print(f"[export] {out}  ({int(sel.sum()):,} particles; "
                  f"PC{pc} in [{edges[b]:.2f}, {edges[b + 1]:.2f}])")
    print(f"[done] -> {args.outdir}")


if __name__ == "__main__":
    main()
