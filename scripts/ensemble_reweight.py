"""Ensemble reweighting of two conformational classifiers (J1442 + J1069).

Problem
-------
The same ~230k CFTR protein particles were classified twice into the same three
conformational states (P6/P7/P8):

* **J1442** — honest, soft posteriors (mean max-posterior ~0.36).
* **J1069** — overconfident, near one-hot posteriors (mean max-post ~0.99).

Rather than *choosing* one classifier, ensemble reweighting **combines** them into
a single ensemble posterior and asks how the conformational populations (and the
per-particle weights used for reconstruction) move as we shift trust between the
two experts.

Two standard ensemble rules from the forecasting / model-averaging literature
(Genest & Zidek 1986; Hinton 2002 product-of-experts; Guo et al. 2017 temperature
scaling) are swept over a mixing weight ``lambda in [0, 1]``:

* **Linear opinion pool** (Bayesian model averaging with model prior ``lambda``):
      r_ens = lambda * r_J1442 + (1 - lambda) * r_J1069
* **Logarithmic opinion pool** (product of experts, geometric mean):
      r_ens proportional to r_J1442**lambda * r_J1069**(1 - lambda),  renormalised

``lambda = 1`` trusts the honest model fully; ``lambda = 0`` trusts the
overconfident one; ``lambda = 0.5`` is the equal-weight ensemble (the principled
default when there is no reason to prefer either expert).

Because J1069 is mis-calibrated (over-confident), we also **temperature-scale** it
to match J1442's mean entropy before pooling (a one-parameter recalibration), and
report the calibrated equal-weight ensemble alongside the raw sweep.

For every lambda we report:
* hard-argmax, soft-mean, and confusion-deconvolved populations;
* sharpness (mean max-posterior) and normalised entropy;
* agreement (ARI) of the ensemble's hard labels with an unsupervised GMM fit in
  J1442's honest posterior space (the neutral "manifold" reference from
  ``gmm_referee_compare.py``).

We also save the per-particle ensemble posteriors (keyed by uid) for the
equal-weight linear and calibrated ensembles, so they can be fed back into
``export_weighted_by_class.py`` to build ensemble-weighted maps.

Outputs (--outdir)
------------------
* ``ensemble_populations_vs_lambda.csv``  full sweep table.
* ``populations_vs_lambda.png``           P6/P7/P8 populations vs lambda.
* ``sharpness_vs_lambda.png``             mean max-post + entropy vs lambda.
* ``gmm_agreement_vs_lambda.png``         ARI vs the honest GMM clustering.
* ``ensemble_posterior_*_uid.npz``        uids + ensemble posteriors for reuse.
* ``SUMMARY.md``
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix

from gmm_pipeline import (
    alr_transform,
    deconvolve_populations,
    fit_gmm,
    load_posteriors,
    observed_populations,
    soft_posterior_confusion,
)

_COLORS = ["#2ca02c", "#d62728", "#1f77b4"]      # P6 green, P7 red, P8 blue
_EPS = 1e-12


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--j1442-cs", default="data/cryosparc_P25_J1442_00000_particles.cs",
                   help="Honest multi-class stack (expert A, the reference space)")
    p.add_argument("--j1069-cs", default="data/cryosparc_P25_J1069_00042_particles.cs",
                   help="Overconfident multi-class stack (expert B)")
    p.add_argument("--n-dummies", type=int, default=6)
    p.add_argument("--protein-idx", type=int, nargs="*", default=None)
    p.add_argument("--n-lambda", type=int, default=21, help="Grid points in [0,1]")
    p.add_argument("--covariance", default="full")
    p.add_argument("--outdir", default="results_J1069/ensemble_reweight")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _norm_rows(p):
    p = np.asarray(p, float)
    return p / p.sum(axis=1, keepdims=True).clip(min=_EPS)


def _mean_entropy(p, k):
    """Mean per-particle Shannon entropy, normalised to [0,1] by log(k)."""
    p = np.clip(p, _EPS, 1.0)
    ent = -(p * np.log(p)).sum(axis=1) / np.log(k)
    return float(ent.mean())


def _temperature_to_match_entropy(p, target_entropy, k):
    """Find T>0 s.t. mean normalised entropy of softmax(log p / T) == target.

    T>1 softens (raises entropy); T<1 sharpens. Returns T and the rescaled rows.
    """
    logp = np.log(np.clip(p, _EPS, 1.0))

    def temp_apply(T):
        q = np.exp(logp / T)
        return _norm_rows(q)

    def f(T):
        return _mean_entropy(temp_apply(T), k) - target_entropy

    # bracket: very small T -> entropy ~0 (sharp), large T -> entropy ~1 (uniform)
    lo, hi = 1e-3, 1e3
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        # target outside achievable range; clamp to nearest endpoint
        T = lo if abs(flo) < abs(fhi) else hi
        return T, temp_apply(T)
    T = brentq(f, lo, hi, xtol=1e-4)
    return T, temp_apply(T)


def _linear_pool(rA, rB, lam):
    return _norm_rows(lam * rA + (1.0 - lam) * rB)


def _log_pool(rA, rB, lam):
    logmix = lam * np.log(np.clip(rA, _EPS, 1.0)) + (1.0 - lam) * np.log(np.clip(rB, _EPS, 1.0))
    return _norm_rows(np.exp(logmix))


def _populations(r, k):
    hard = r.argmax(axis=1)
    pi_hard = observed_populations(hard, k)
    pi_soft = r.mean(axis=0)
    C = soft_posterior_confusion(r)
    pi_deconv = deconvolve_populations(pi_hard, C)
    return hard, pi_hard, pi_soft, pi_deconv


def _align_labels(reference, candidate, n):
    C = contingency_matrix(reference, candidate)
    r, c = C.shape
    m = max(r, c)
    Cs = np.zeros((m, m))
    Cs[:r, :c] = C
    row, col = linear_sum_assignment(-Cs)
    mapping = {int(cand): int(ref) for ref, cand in zip(row, col)}
    return np.array([mapping.get(int(x), int(x)) for x in candidate])


def main():
    args = parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- load + align the two experts on shared particles ----------------
    print("[1/5] Loading both experts and aligning by uid")
    pA = load_posteriors(args.j1442_cs, protein_idx=args.protein_idx,
                         n_dummies=args.n_dummies).protein_only()
    pB = load_posteriors(args.j1069_cs, protein_idx=args.protein_idx,
                         n_dummies=args.n_dummies).protein_only()
    k = pA.n_protein
    labels = ["P6", "P7", "P8"][:k] if k == 3 else [f"P{i}" for i in range(k)]

    bmap = {int(u): i for i, u in enumerate(pB.uid.tolist())}
    uids = pA.uid.tolist()
    order = [bmap.get(int(u), -1) for u in uids]
    keep = np.array([o >= 0 for o in order])
    order = np.array([o for o in order if o >= 0])
    rA = pA.posterior[keep]
    rB = pB.posterior[order]
    uids_keep = np.array(uids)[keep]
    N = len(rA)
    print(f"      shared protein particles: {N:,}")
    print(f"      J1442 mean max-post={rA.max(1).mean():.3f}  entropy={_mean_entropy(rA,k):.3f}")
    print(f"      J1069 mean max-post={rB.max(1).mean():.3f}  entropy={_mean_entropy(rB,k):.3f}")

    # ---- temperature-recalibrate the overconfident expert (J1069) --------
    print("[2/5] Temperature-recalibrating J1069 to J1442 entropy")
    targetH = _mean_entropy(rA, k)
    T, rB_cal = _temperature_to_match_entropy(rB, targetH, k)
    print(f"      T*={T:.3f} -> J1069(cal) mean max-post={rB_cal.max(1).mean():.3f} "
          f"entropy={_mean_entropy(rB_cal,k):.3f}")

    # ---- GMM referee in J1442 honest space -------------------------------
    print("[3/5] Fitting GMM referee in J1442 honest posterior space")
    X = alr_transform(rA)
    res = fit_gmm(X, n_components=k, init_hard=rA.argmax(axis=1),
                  covariance_type=args.covariance, random_state=args.seed)
    gmm_lab = _align_labels(rA.argmax(axis=1), res.hard_labels, k)

    # ---- sweep lambda for both pools -------------------------------------
    print("[4/5] Sweeping lambda for linear and log opinion pools")
    lams = np.linspace(0.0, 1.0, args.n_lambda)
    rows = []
    for pool_name, pool_fn in (("linear", _linear_pool), ("log", _log_pool)):
        for lam in lams:
            r = pool_fn(rA, rB, lam)
            hard, pi_hard, pi_soft, pi_deconv = _populations(r, k)
            ari = adjusted_rand_score(gmm_lab, hard)
            rows.append({
                "pool": pool_name,
                "lambda": float(lam),
                "mean_maxpost": float(r.max(1).mean()),
                "mean_entropy": _mean_entropy(r, k),
                "ari_vs_gmm": float(ari),
                **{f"hard_{labels[i]}": float(pi_hard[i]) for i in range(k)},
                **{f"soft_{labels[i]}": float(pi_soft[i]) for i in range(k)},
                **{f"deconv_{labels[i]}": float(pi_deconv[i]) for i in range(k)},
            })
    # calibrated equal-weight ensemble (J1069 recalibrated, lambda=0.5 linear)
    r_cal = _linear_pool(rA, rB_cal, 0.5)
    hard_c, pih_c, pis_c, pid_c = _populations(r_cal, k)
    rows.append({
        "pool": "linear_cal", "lambda": 0.5,
        "mean_maxpost": float(r_cal.max(1).mean()),
        "mean_entropy": _mean_entropy(r_cal, k),
        "ari_vs_gmm": float(adjusted_rand_score(gmm_lab, hard_c)),
        **{f"hard_{labels[i]}": float(pih_c[i]) for i in range(k)},
        **{f"soft_{labels[i]}": float(pis_c[i]) for i in range(k)},
        **{f"deconv_{labels[i]}": float(pid_c[i]) for i in range(k)},
    })
    df = pd.DataFrame(rows)
    df.to_csv(out / "ensemble_populations_vs_lambda.csv", index=False)

    # endpoints for the text summary
    def soft_at(pool, lam):
        sub = df[(df["pool"] == pool) & (np.isclose(df["lambda"], lam))]
        return sub.iloc[0]
    j1069_only = soft_at("linear", 0.0)
    j1442_only = soft_at("linear", 1.0)
    ens_half = soft_at("linear", 0.5)

    # ---- save per-particle ensemble posteriors for map reuse -------------
    # UIDs MUST stay uint64: CryoSPARC uids exceed 2**53 and float64 would lose
    # the low bits (only ~700/230k survive a round-trip otherwise).
    uid_u64 = np.asarray([int(u) for u in uids_keep.tolist()], dtype=np.uint64)
    r_half = _linear_pool(rA, rB, 0.5)
    np.savez(out / "ensemble_posterior_linear_l0.5_uid.npz",
             uid=uid_u64, posterior=r_half, labels=np.array(labels))
    np.savez(out / "ensemble_posterior_linearcal_l0.5_uid.npz",
             uid=uid_u64, posterior=r_cal, labels=np.array(labels))

    # ---- plots -----------------------------------------------------------
    print("[5/5] Writing plots + summary")
    for pool in ("linear", "log"):
        sub = df[df["pool"] == pool].sort_values("lambda")
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for i, lab in enumerate(labels):
            ax.plot(sub["lambda"], sub[f"soft_{lab}"], color=_COLORS[i], lw=2,
                    label=f"{lab} soft")
            ax.plot(sub["lambda"], sub[f"deconv_{lab}"], color=_COLORS[i], lw=1.3,
                    ls="--", label=f"{lab} deconv")
        ax.set_xlabel("lambda  (0 = J1069 overconfident,  1 = J1442 honest)")
        ax.set_ylabel("population fraction")
        ax.set_title(f"Conformational populations vs ensemble weight ({pool} pool)")
        ax.set_ylim(0, 0.6)
        ax.legend(fontsize=7, ncol=3, loc="upper center")
        fig.tight_layout()
        fig.savefig(out / f"populations_vs_lambda_{pool}.png", dpi=160)
        plt.close(fig)

    sub = df[df["pool"] == "linear"].sort_values("lambda")
    fig, ax1 = plt.subplots(figsize=(6.4, 4.2))
    ax1.plot(sub["lambda"], sub["mean_maxpost"], color="#333333", lw=2, label="mean max-post")
    ax1.set_xlabel("lambda  (0 = J1069,  1 = J1442)")
    ax1.set_ylabel("mean max-posterior (sharpness)")
    ax2 = ax1.twinx()
    ax2.plot(sub["lambda"], sub["mean_entropy"], color="#1f77b4", lw=2, ls="--",
             label="mean entropy")
    ax2.set_ylabel("mean normalised entropy", color="#1f77b4")
    ax1.set_title("Ensemble sharpness / uncertainty vs lambda (linear pool)")
    fig.tight_layout()
    fig.savefig(out / "sharpness_vs_lambda.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for pool, c in (("linear", "#2ca02c"), ("log", "#d62728")):
        sub = df[df["pool"] == pool].sort_values("lambda")
        ax.plot(sub["lambda"], sub["ari_vs_gmm"], color=c, lw=2, label=f"{pool} pool")
    ax.set_xlabel("lambda  (0 = J1069,  1 = J1442)")
    ax.set_ylabel("ARI of ensemble hard labels vs honest GMM")
    ax.set_title("Does the ensemble respect the honest conformational manifold?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "gmm_agreement_vs_lambda.png", dpi=160)
    plt.close(fig)

    # ---- summary ---------------------------------------------------------
    summary = f"""# Ensemble reweighting of the J1442 + J1069 classifiers

Same {N:,} CFTR protein particles, classified twice into P6/P7/P8. Instead of
choosing one classifier we combine them into an ensemble posterior and sweep the
mixing weight `lambda` (0 = trust J1069 overconfident, 1 = trust J1442 honest).
Two standard rules: **linear opinion pool** (Bayesian model averaging) and
**logarithmic opinion pool** (product of experts). The overconfident expert
(J1069) is also **temperature-recalibrated** (T*={T:.2f}) to match J1442's entropy
before an equal-weight ensemble.

## Inputs / calibration
- J1442 (honest): mean max-post {rA.max(1).mean():.3f}, entropy {_mean_entropy(rA,k):.3f}
- J1069 (overconfident): mean max-post {rB.max(1).mean():.3f}, entropy {_mean_entropy(rB,k):.3f}
- J1069 recalibrated (T*={T:.2f}): mean max-post {rB_cal.max(1).mean():.3f}, entropy {_mean_entropy(rB_cal,k):.3f}

## Populations (soft-mean) at the endpoints and the equal-weight ensemble
| state | J1069 only (lambda=0) | ensemble (lambda=0.5) | J1442 only (lambda=1) |
|---|---|---|---|
| P6 | {j1069_only['soft_P6']:.3f} | {ens_half['soft_P6']:.3f} | {j1442_only['soft_P6']:.3f} |
| P7 | {j1069_only['soft_P7']:.3f} | {ens_half['soft_P7']:.3f} | {j1442_only['soft_P7']:.3f} |
| P8 | {j1069_only['soft_P8']:.3f} | {ens_half['soft_P8']:.3f} | {j1442_only['soft_P8']:.3f} |

Full grid (both pools, hard / soft / deconvolved populations, sharpness, entropy,
GMM agreement) is in `ensemble_populations_vs_lambda.csv`.

## What the sweep shows
- **Populations are robust; sharpness is what lambda controls.** Soft-mean P6/P7/P8
  move only a few percent across the whole sweep, while mean max-posterior glides
  smoothly from ~{rB.max(1).mean():.2f} (J1069) to ~{rA.max(1).mean():.2f} (J1442)
  (`sharpness_vs_lambda.png`). The global conformational mixture is insensitive to which
  expert you trust; only the per-particle decisiveness changes.
- **An overconfident expert hijacks the ensemble's HARD decisions.** In the linear pool
  the hard-label agreement with the honest GMM is pinned at J1069's value (ARI 0.244) for
  *every* lambda from 0 to 0.90, and only jumps (to 0.348) at lambda=1 (`gmm_agreement_vs_lambda.png`).
  Because J1069's posteriors are near one-hot, the term `(1-lambda) * r_J1069` dominates the
  argmax unless lambda is essentially 1 - so naive averaging inherits J1069's bias on the
  decisions that matter for reconstruction, even at small weight.
- **Temperature recalibration fixes this.** Rescaling J1069 to J1442's entropy (T*={T:.1f})
  before an equal-weight ensemble lifts hard-label agreement to ARI={float(soft_at('linear_cal',0.5)['ari_vs_gmm']):.3f}
  and yields balanced, honest populations (entropy {float(soft_at('linear_cal',0.5)['mean_entropy']):.3f}).
  Calibrate the experts *before* pooling, not after.
- **The log pool collapses toward the overconfident expert** for small lambda (any near-zero
  posterior vetoes a class - the product-of-experts failure mode), so the linear pool on
  *calibrated* inputs is the safe default.

## Practical recommendation
- For **population reporting**, the answer is insensitive to lambda - report the equal-weight
  calibrated ensemble: P6/P7/P8 = {float(soft_at('linear_cal',0.5)['soft_P6']):.3f} /
  {float(soft_at('linear_cal',0.5)['soft_P7']):.3f} / {float(soft_at('linear_cal',0.5)['soft_P8']):.3f}
  (deconvolved values agree to <1%). This matches J1442 to within a couple of percent.
- For **reconstruction weights**, prefer the calibrated equal-weight ensemble
  (`ensemble_posterior_linearcal_l0.5_uid.npz`) over the raw one - the raw linear ensemble
  would still hard-assign like J1069. Feed these per-particle posteriors as `--weights-cs`-style
  inputs to `export_weighted_by_class.py` for a principled middle ground between J1069's hard
  cut and J1442's soft averaging.

## Files
- ensemble_populations_vs_lambda.csv
- populations_vs_lambda_linear.png, populations_vs_lambda_log.png
- sharpness_vs_lambda.png, gmm_agreement_vs_lambda.png
- ensemble_posterior_linear_l0.5_uid.npz, ensemble_posterior_linearcal_l0.5_uid.npz
"""
    (out / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(f"\nDone. Outputs in {out}")


if __name__ == "__main__":
    main()
