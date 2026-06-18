"""Local follow-up analyses that do NOT require raw particle images.

Answers two questions from .cs metadata you already have:

  1. Class occupancy & recoverable P8 signal
       - hard particle count for ALL classes (dummies + protein)
       - dummy occupancy (how much mass sits in dummy classes)
       - soft "effective N" per protein class (sum of responsibilities)
       - P8 effective N as a function of the sharpening exponent beta
         (mirrors export_weighted_by_class.py's weighting)

  2. Is the P6/P7 pair continuous or two discrete states?
       - project protein posteriors onto the P6-vs-P7 discriminant axis
       - fit a 1-component vs 2-component GMM to that 1-D axis
       - compare BIC: 1-comp wins  -> continuous (one basin, over-split)
                      2-comp wins  -> genuinely two discrete states

This deliberately does NOT recompute things already produced by
  - scripts/compare_maps.py        (map CC / FSC / difference maps)
  - scripts/posterior_diagnostics.py (entropy / violin / max-posterior)
  - run_pipeline.py                (confusion matrices / corrected populations)

Run:
    python scripts/local_followups.py \
        --cs data/cryosparc_P25_J1442_00000_particles.cs \
        --n-dummies 6 --outdir results_J1442/local_followups
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gmm_pipeline import load_posteriors, alr_transform  # noqa: E402

try:
    from sklearn.mixture import GaussianMixture
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"scikit-learn is required: {exc}")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def effective_n_by_beta(resp: np.ndarray, betas) -> dict:
    """resp: (N, K) protein responsibilities (rows sum to 1).

    Returns {beta: (K,) effective N} where weights are
        w_nk = r_nk^beta / sum_j r_nj^beta   then summed over particles.
    """
    out = {}
    for b in betas:
        rb = np.power(resp, b)
        w = rb / rb.sum(axis=1, keepdims=True).clip(min=1e-12)
        out[float(b)] = w.sum(axis=0)
    return out


def continuity_test(y: np.ndarray):
    """1-D continuity test on axis values y.

    Fits 1- and 2-component Gaussian mixtures; returns dict with BICs,
    the verdict, and (if bimodal) the separation of the two means in SDs.
    """
    y = y.reshape(-1, 1)
    g1 = GaussianMixture(n_components=1, covariance_type="full",
                         random_state=0).fit(y)
    g2 = GaussianMixture(n_components=2, covariance_type="full",
                         random_state=0, n_init=3).fit(y)
    bic1, bic2 = g1.bic(y), g2.bic(y)

    means = g2.means_.ravel()
    sds = np.sqrt(g2.covariances_.ravel())
    weights = g2.weights_.ravel()
    # standardized separation between the two component means
    pooled_sd = np.sqrt((weights * sds**2).sum())
    sep = abs(means[0] - means[1]) / max(pooled_sd, 1e-9)

    # 2 components is only meaningfully "discrete" if BIC clearly prefers it
    # AND the two modes are well separated (>~2 SD) and both non-trivial.
    bic_prefers_2 = (bic1 - bic2) > 10.0
    well_separated = sep > 2.0
    both_populated = weights.min() > 0.10
    discrete = bool(bic_prefers_2 and well_separated and both_populated)

    return {
        "bic_1comp": float(bic1),
        "bic_2comp": float(bic2),
        "delta_bic_1minus2": float(bic1 - bic2),
        "mode_separation_sd": float(sep),
        "min_component_weight": float(weights.min()),
        "verdict": "discrete (two states)" if discrete
        else "continuous (one basin / over-split)",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cs", required=True, help="cryosparc_*_particles.cs")
    ap.add_argument("--n-dummies", type=int, default=6)
    ap.add_argument("--betas", type=float, nargs="+",
                    default=[1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--pair", nargs=2, default=["P6", "P7"],
                    help="which two protein classes to test for continuity")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading {args.cs}")
    post = load_posteriors(args.cs, n_dummies=args.n_dummies)
    N = len(post.uid)
    K_full = post.n_classes
    protein_idx = post.protein_idx
    dummy_idx = post.dummy_idx
    prot_labels = [f"P{args.n_dummies + i}" for i in range(len(protein_idx))]
    print(f"      N={N:,}  K_full={K_full}  protein={prot_labels}")

    # -------------------------------------------------------------------
    # 1. Class occupancy (all classes) + dummy occupancy
    # -------------------------------------------------------------------
    hard_full = post.posterior.argmax(axis=1)
    counts_full = np.bincount(hard_full, minlength=K_full)
    dummy_mask = np.isin(hard_full, dummy_idx)
    dummy_count = int(dummy_mask.sum())
    dummy_frac = dummy_count / N

    print(f"[2/4] Class occupancy (hard argmax over all {K_full} classes)")
    occ_rows = []
    for i in range(K_full):
        kind = "dummy" if i in dummy_idx else "protein"
        lbl = ("D%d" % i) if i in dummy_idx else prot_labels[list(protein_idx).index(i)]
        occ_rows.append({
            "class_index": i, "label": lbl, "kind": kind,
            "hard_count": int(counts_full[i]),
            "hard_fraction": counts_full[i] / N,
        })
    print(f"      dummy occupancy: {dummy_count:,} particles "
          f"({dummy_frac*100:.1f}% of all particles)")

    # -------------------------------------------------------------------
    # 2. Soft effective N per protein class (responsibilities)
    # -------------------------------------------------------------------
    # protein responsibilities for ALL particles, renormalized over protein
    resp_all = post.posterior[:, protein_idx]
    resp_all = resp_all / resp_all.sum(axis=1, keepdims=True).clip(min=1e-12)
    soft_eff_all = resp_all.sum(axis=0)  # (K_protein,)

    # protein responsibilities restricted to protein-hard particles
    prot_hard_mask = np.isin(hard_full, protein_idx)
    resp_ph = post.posterior[prot_hard_mask][:, protein_idx]
    resp_ph = resp_ph / resp_ph.sum(axis=1, keepdims=True).clip(min=1e-12)
    soft_eff_ph = resp_ph.sum(axis=0)

    for j, lbl in enumerate(prot_labels):
        print(f"      {lbl}: soft eff-N (all)={soft_eff_all[j]:,.0f}  "
              f"(protein-hard only)={soft_eff_ph[j]:,.0f}")

    # -------------------------------------------------------------------
    # 3. P8 (and all protein) effective N vs sharpening beta
    # -------------------------------------------------------------------
    eff_by_beta = effective_n_by_beta(resp_all, args.betas)
    print("[3/4] Effective N per protein class vs sharpening beta")
    for b in args.betas:
        vals = ", ".join(f"{l}={v:,.0f}" for l, v in
                         zip(prot_labels, eff_by_beta[float(b)]))
        print(f"      beta={b:g}: {vals}")

    # -------------------------------------------------------------------
    # 4. Continuity test on the requested pair (default P6 vs P7)
    # -------------------------------------------------------------------
    pa, pb = args.pair
    if pa not in prot_labels or pb not in prot_labels:
        raise SystemExit(f"--pair {pa} {pb} not in protein labels {prot_labels}")
    ia, ib = prot_labels.index(pa), prot_labels.index(pb)

    # ALR transform of protein posteriors, then the discriminant axis is
    # simply log(P_a / P_b). Use only particles that are confidently in the
    # {pa, pb} subspace (their summed posterior dominates) to avoid dummy noise.
    prot_post = resp_all  # already renormalized over protein
    ratio_axis = np.log(prot_post[:, ia].clip(1e-9) / prot_post[:, ib].clip(1e-9))
    # restrict to particles where pa+pb carry most of the protein mass
    pair_mass = prot_post[:, ia] + prot_post[:, ib]
    sel = pair_mass > (1.0 / len(protein_idx))  # above uniform share
    y = ratio_axis[sel]
    print(f"[4/4] Continuity test on {pa} vs {pb} axis  "
          f"(N_used={len(y):,} of {N:,})")
    ct = continuity_test(y)
    for k, v in ct.items():
        print(f"      {k}: {v}")

    # -------------------------------------------------------------------
    # Write CSVs
    # -------------------------------------------------------------------
    with open(out / "class_occupancy.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(occ_rows[0].keys()))
        w.writeheader()
        w.writerows(occ_rows)

    with open(out / "protein_effective_N.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "soft_effN_all_particles",
                    "soft_effN_protein_hard"] +
                   [f"effN_beta{b:g}" for b in args.betas])
        for j, lbl in enumerate(prot_labels):
            w.writerow([lbl, f"{soft_eff_all[j]:.2f}", f"{soft_eff_ph[j]:.2f}"] +
                       [f"{eff_by_beta[float(b)][j]:.2f}" for b in args.betas])

    with open(out / f"continuity_{pa}_vs_{pb}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value"])
        for k, v in ct.items():
            w.writerow([k, v])

    # -------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------
    # (a) occupancy bar chart (dummies grey, protein coloured)
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = np.arange(K_full)
    colors = ["0.7" if i in dummy_idx else "C0" for i in range(K_full)]
    ax.bar(xs, counts_full, color=colors)
    ax.set_xticks(xs)
    ax.set_xticklabels([r["label"] for r in occ_rows], rotation=45, ha="right")
    ax.set_ylabel("hard-assigned particles")
    ax.set_title(f"Class occupancy (grey = dummy, blue = protein)\n"
                 f"dummy occupancy = {dummy_frac*100:.1f}% of all particles")
    fig.tight_layout()
    fig.savefig(out / "class_occupancy.png", dpi=150)
    plt.close(fig)

    # (b) effective N vs beta for protein classes
    fig, ax = plt.subplots(figsize=(7, 4))
    for j, lbl in enumerate(prot_labels):
        ax.plot(args.betas, [eff_by_beta[float(b)][j] for b in args.betas],
                marker="o", label=lbl)
    ax.set_xlabel("sharpening exponent beta")
    ax.set_ylabel("effective N (sum of weights)")
    ax.set_title("Recoverable particles per class vs weighting sharpness\n"
                 "(beta=1 unbiased soft; larger beta -> hard argmax)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "effective_N_vs_beta.png", dpi=150)
    plt.close(fig)

    # (c) continuity histogram + fitted modes
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(y, bins=80, density=True, color="0.8", edgecolor="none")
    g2 = GaussianMixture(n_components=2, covariance_type="full",
                         random_state=0, n_init=3).fit(y.reshape(-1, 1))
    xs = np.linspace(y.min(), y.max(), 400).reshape(-1, 1)
    dens = np.exp(g2.score_samples(xs))
    ax.plot(xs.ravel(), dens, "C3", lw=1.8, label="2-component fit")
    ax.axvline(0.0, color="0.5", ls="--", lw=1, label=f"equal {pa}/{pb}")
    ax.set_xlabel(f"log( {pa} / {pb} )  [discriminant axis]")
    ax.set_ylabel("density")
    ax.set_title(f"{pa} vs {pb} continuity test\nverdict: {ct['verdict']}  "
                 f"(ΔBIC={ct['delta_bic_1minus2']:.0f}, "
                 f"sep={ct['mode_separation_sd']:.2f} SD)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"continuity_{pa}_vs_{pb}.png", dpi=150)
    plt.close(fig)

    print(f"\nDone. Outputs in {out}:")
    print("  class_occupancy.csv / .png")
    print("  protein_effective_N.csv / effective_N_vs_beta.png")
    print(f"  continuity_{pa}_vs_{pb}.csv / .png")


if __name__ == "__main__":
    main()
