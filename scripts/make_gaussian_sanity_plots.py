"""Gaussian-fit sanity-check plots (Hunt 06/08/26 notes).

Reproduces the pipeline GMM fit for a CryoSPARC dataset and renders:

* 1-D Gaussian sanity histograms along each protein posterior axis (P6/P7/P8),
  broken down by class subset (e.g. 6, 8, 6+8, 6+7+8) with per-class Gaussian
  fits, the component mixture, and the back-projected fitted GMM overlaid.
* 2-D Gaussian sanity plots in raw posterior space: empirical per-class
  ellipses (solid) vs the fitted GMM back-projected from ALR space (dashed).

Usage
-----
    python make_gaussian_sanity_plots.py --cs cryosparc_P25_J1442_00000_particles.cs \
        --n-dummies 6 --outdir results_J1442
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gmm_pipeline import alr_transform, fit_gmm, load_posteriors
from gmm_pipeline.plots import (
    plot_axis_gaussian_sanity,
    plot_posterior_space_gmm,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cs", required=True, help="CryoSPARC *_particles.cs file")
    p.add_argument("--n-dummies", type=int, default=6)
    p.add_argument("--protein-idx", type=int, nargs="*", default=None)
    p.add_argument("--covariance", default="full")
    p.add_argument("--outdir", default="results")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.outdir) / "sanity"
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading {args.cs}")
    post = load_posteriors(args.cs, protein_idx=args.protein_idx,
                           n_dummies=args.n_dummies)
    prot = post.protein_only()
    labels = [f"P{int(c)}" for c in post.protein_idx]
    K = prot.n_protein
    print(f"      N_protein={len(prot.uid):,}  K_protein={K}  labels={labels}")

    print("[2/4] Fitting GMM in ALR space (same as pipeline)")
    X = alr_transform(prot.posterior)
    res = fit_gmm(X, n_components=K, init_hard=prot.hard_class,
                  covariance_type=args.covariance, random_state=args.seed)
    print(f"      converged={res.converged}  BIC={res.bic:.1f}")

    alr_ref = -1  # alr_transform default reference column (last protein class)

    print("[3/4] 1-D Gaussian sanity histograms (per protein axis)")
    for axis in range(K):
        fname = out / f"gaussian_sanity_1d_{labels[axis]}.png"
        plot_axis_gaussian_sanity(
            prot.posterior, prot.hard_class, labels,
            axis=axis, gmm=res.model, alr_ref=alr_ref,
            out=fname, random_state=args.seed,
        )
        print(f"      saved {fname.name}")

    print("[4/4] 2-D Gaussian sanity plot in raw posterior space")
    fname2d = out / "gaussian_sanity_2d_posterior.png"
    plot_posterior_space_gmm(
        prot.posterior, prot.hard_class, labels,
        gmm=res.model, alr_ref=alr_ref,
        out=fname2d, random_state=args.seed,
    )
    print(f"      saved {fname2d.name}")
    print("done.")


if __name__ == "__main__":
    main()
