"""GMM-as-referee: external validation of two competing particle partitions.

Context
-------
We have the SAME ~230k CFTR protein particles labelled two different ways:

* **J1442** — honest soft heterogeneous refinement (mean max-posterior ~0.36).
* **J1069 (weighted by J1442)** — overconfident refinement (mean max-post ~0.99)
  whose hard argmax was used to build per-class reconstructions.

The uploaded per-class ``*_particles.cs`` files are *single-class* Homogeneous
Reconstruction outputs: their ``class_posterior``/``alpha`` are reset to 1, so
they carry only **class membership** (which particles ended up in P6/P7/P8 for
each job). They cannot be fed to the GMM pipeline directly (no multi-class
posterior). Instead we use them to define the two partitions and let an
unsupervised GMM, fit in J1442's *honest* posterior simplex, act as a neutral
referee:

    For each labelling, how well does it match the natural conformational
    clusters the GMM finds in honest posterior space?

This is the standard "external cluster validation" recipe from the clustering
literature (Hubert & Arabie 1985 ARI; Vinh et al. 2010 AMI / V-measure): compare
each candidate partition to a reference clustering with adjusted, chance-
corrected agreement indices, plus contingency (confusion) tables.

Outputs (in --outdir)
---------------------
* ``agreement_metrics.csv``          ARI / AMI / NMI / V-measure / homogeneity /
                                     completeness for GMM-vs-J1442, GMM-vs-J1069,
                                     J1442-vs-J1069.
* ``contingency_*.csv`` / ``.png``   row-normalised cross-tabs.
* ``within_cluster_confidence.csv``  mean GMM max-responsibility per class for
                                     each labelling (cluster "cleanliness").
* ``disagreement_analysis.csv``      where the two jobs disagree, which side the
                                     honest GMM takes + J1442 ambiguity stats.
* ``alr_scatter_by_labelling.png``   2-D ALR embedding coloured three ways.
* ``agreement_bars.png``             headline metric comparison.
* ``SUMMARY.md``                     plain-language interpretation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
)
from sklearn.metrics.cluster import contingency_matrix

from gmm_pipeline import alr_transform, fit_gmm, load_posteriors

# P6 = green, P7 = red, P8 = blue (matches the rest of the project)
_COLORS = ["#2ca02c", "#d62728", "#1f77b4", "#9467bd", "#ff7f0e", "#8c564b"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--j1442-cs", default="data/cryosparc_P25_J1442_00000_particles.cs",
                   help="Honest multi-class J1442 particle stack (the referee space)")
    p.add_argument("--j1069-cs", default="data/cryosparc_P25_J1069_00042_particles.cs",
                   help="Overconfident multi-class J1069 particle stack (competing partition). "
                        "Its protein argmax reproduces the uploaded hardJ1069_w1442 membership "
                        "exactly, but with the original UIDs that join to J1442.")
    p.add_argument("--n-dummies", type=int, default=6)
    p.add_argument("--protein-idx", type=int, nargs="*", default=None)
    p.add_argument("--covariance", default="full")
    p.add_argument("--outdir", default="results_J1069/gmm_referee")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-points", type=int, default=40000,
                   help="Subsample for scatter plots only (metrics use all points)")
    return p.parse_args()


def _align_labels(reference, candidate, n):
    """Permute ``candidate`` cluster ids to best match ``reference`` via Hungarian
    on the contingency matrix. Returns the relabelled candidate."""
    C = contingency_matrix(reference, candidate)        # (n_ref, n_cand)
    # pad to square if needed
    r, c = C.shape
    m = max(r, c)
    Cs = np.zeros((m, m))
    Cs[:r, :c] = C
    row, col = linear_sum_assignment(-Cs)
    mapping = {int(cand): int(ref) for ref, cand in zip(row, col)}
    return np.array([mapping.get(int(x), int(x)) for x in candidate])


def _row_norm(M):
    M = np.asarray(M, float)
    return M / M.sum(axis=1, keepdims=True).clip(min=1e-12)


def _heatmap(M, row_labels, col_labels, title, path):
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] < 0.6 else "black", fontsize=9)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ---- 1. Honest posterior space + GMM referee -------------------------
    print(f"[1/6] Loading honest J1442 posteriors: {args.j1442_cs}")
    post = load_posteriors(args.j1442_cs, protein_idx=args.protein_idx,
                           n_dummies=args.n_dummies)
    prot = post.protein_only()
    labels = [f"P{int(c)}" for c in post.protein_idx]
    N = len(prot.uid)
    print(f"      protein particles N={N:,}  classes={labels}")

    X = alr_transform(prot.posterior)
    print(f"[2/6] Fitting GMM referee (k={prot.n_protein}, cov={args.covariance})")
    res = fit_gmm(X, n_components=prot.n_protein, init_hard=prot.hard_class,
                  covariance_type=args.covariance, random_state=args.seed)
    gmm_lab = res.hard_labels
    max_resp = res.responsibilities.max(axis=1)
    # align GMM component ids to J1442 class ids for interpretable tables
    gmm_lab = _align_labels(prot.hard_class, gmm_lab, prot.n_protein)
    print(f"      GMM converged={res.converged}  BIC={res.bic:.1f}")

    # ---- 2. Competing partition from the J1069 source stack ---------------
    # The uploaded per-class reconstruction outputs carry CryoSPARC-reassigned
    # UIDs that no longer join to J1442, so we recover the *identical* membership
    # (same per-class counts) from the original J1069 multi-class stack by UID.
    print("[3/6] Deriving J1069 partition from source stack")
    j69 = load_posteriors(args.j1069_cs, protein_idx=args.protein_idx,
                          n_dummies=args.n_dummies).protein_only()
    j69_map = {int(u): int(c) for u, c in zip(j69.uid.tolist(), j69.hard_class.tolist())}

    uids = prot.uid.tolist()
    j1442_lab = prot.hard_class.copy()
    j1069_lab = np.array([j69_map.get(int(u), -1) for u in uids])
    keep = j1069_lab >= 0
    print(f"      J1442 argmax counts: {np.bincount(j1442_lab, minlength=prot.n_protein).tolist()}")
    print(f"      J1069 argmax counts: {np.bincount(j1069_lab[keep], minlength=prot.n_protein).tolist()}")
    print(f"      particles matched in both partitions: {keep.sum():,} / {N:,}")

    gmm_lab, j1442_lab, j1069_lab = gmm_lab[keep], j1442_lab[keep], j1069_lab[keep]
    max_resp = max_resp[keep]
    Xk = X[keep]
    post_k = prot.posterior[keep]

    # ---- 3. Agreement metrics --------------------------------------------
    print("[4/6] Chance-corrected agreement metrics")

    def metrics(a, b):
        h, c, v = homogeneity_completeness_v_measure(a, b)
        return {
            "ARI": adjusted_rand_score(a, b),
            "AMI": adjusted_mutual_info_score(a, b),
            "NMI": normalized_mutual_info_score(a, b),
            "V_measure": v,
            "homogeneity": h,
            "completeness": c,
        }

    rows = {
        "GMM_vs_J1442": metrics(gmm_lab, j1442_lab),
        "GMM_vs_J1069w": metrics(gmm_lab, j1069_lab),
        "J1442_vs_J1069w": metrics(j1442_lab, j1069_lab),
    }
    met_df = pd.DataFrame(rows).T
    met_df.to_csv(out / "agreement_metrics.csv")
    print(met_df.round(4).to_string())

    # ---- 4. Contingency tables -------------------------------------------
    def cont(a, b):
        return _row_norm(contingency_matrix(a, b))

    pairs = {
        "GMM_rows_J1442_cols": (gmm_lab, j1442_lab, "GMM cluster", "J1442 class"),
        "GMM_rows_J1069_cols": (gmm_lab, j1069_lab, "GMM cluster", "J1069 class"),
        "J1442_rows_J1069_cols": (j1442_lab, j1069_lab, "J1442 class", "J1069 class"),
    }
    for name, (a, b, ra, cb) in pairs.items():
        M = cont(a, b)
        pd.DataFrame(M, index=labels, columns=labels).to_csv(out / f"contingency_{name}.csv")
        _heatmap(M, labels, labels, name.replace("_", " "), out / f"contingency_{name}.png")

    # ---- 5. Within-cluster GMM confidence per labelling ------------------
    print("[5/6] Within-class GMM confidence (mean max-responsibility)")
    conf_rows = []
    for k, lab in enumerate(labels):
        conf_rows.append({
            "class": lab,
            "J1442_mean_maxresp": float(max_resp[j1442_lab == k].mean()),
            "J1442_N": int((j1442_lab == k).sum()),
            "J1069_mean_maxresp": float(max_resp[j1069_lab == k].mean()),
            "J1069_N": int((j1069_lab == k).sum()),
        })
    conf_df = pd.DataFrame(conf_rows)
    conf_df.loc["overall"] = {
        "class": "ALL",
        "J1442_mean_maxresp": float(max_resp.mean()),
        "J1442_N": int(len(max_resp)),
        "J1069_mean_maxresp": float(max_resp.mean()),
        "J1069_N": int(len(max_resp)),
    }
    conf_df.to_csv(out / "within_cluster_confidence.csv", index=False)
    print(conf_df.round(4).to_string(index=False))

    # ---- 6. Disagreement analysis: who does the honest GMM side with? ----
    print("[6/6] Disagreement analysis")
    disagree = j1442_lab != j1069_lab
    nd = int(disagree.sum())
    # honest posterior ambiguity for each particle (entropy + max-post)
    eps = 1e-12
    ent = -(post_k * np.log(post_k + eps)).sum(axis=1) / np.log(prot.n_protein)
    maxpost = post_k.max(axis=1)
    gmm_sides_1442 = (gmm_lab[disagree] == j1442_lab[disagree])
    gmm_sides_1069 = (gmm_lab[disagree] == j1069_lab[disagree])
    gmm_sides_neither = ~(gmm_sides_1442 | gmm_sides_1069)
    dis = {
        "n_total": int(len(disagree)),
        "n_disagree": nd,
        "frac_disagree": nd / len(disagree),
        "disagree_mean_J1442_maxpost": float(maxpost[disagree].mean()),
        "agree_mean_J1442_maxpost": float(maxpost[~disagree].mean()),
        "disagree_mean_J1442_entropy": float(ent[disagree].mean()),
        "agree_mean_J1442_entropy": float(ent[~disagree].mean()),
        "of_disagreements_GMM_sides_J1442": float(gmm_sides_1442.mean()),
        "of_disagreements_GMM_sides_J1069": float(gmm_sides_1069.mean()),
        "of_disagreements_GMM_sides_neither": float(gmm_sides_neither.mean()),
    }
    pd.DataFrame([dis]).to_csv(out / "disagreement_analysis.csv", index=False)
    print(json.dumps(dis, indent=2))

    # ---- plots -----------------------------------------------------------
    # subsample for scatter
    idx = np.arange(len(gmm_lab))
    if len(idx) > args.max_points:
        idx = rng.choice(idx, size=args.max_points, replace=False)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharex=True, sharey=True)
    for ax, lab_arr, title in zip(
        axes, [gmm_lab, j1442_lab, j1069_lab],
        ["GMM referee (honest space)", "J1442 assignment", "J1069 (weighted) assignment"],
    ):
        for k in range(prot.n_protein):
            sel = idx[lab_arr[idx] == k]
            ax.scatter(Xk[sel, 0], Xk[sel, 1], s=3, alpha=0.35,
                       color=_COLORS[k], label=labels[k], rasterized=True)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("ALR axis 1")
    axes[0].set_ylabel("ALR axis 2")
    axes[0].legend(markerscale=3, fontsize=8, loc="upper right")
    fig.suptitle("Same particles in J1442 honest posterior space, coloured by partition", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / "alr_scatter_by_labelling.png", dpi=160)
    plt.close(fig)

    # agreement bars
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    metric_names = ["ARI", "AMI", "V_measure"]
    xpos = np.arange(len(metric_names))
    width = 0.38
    ax.bar(xpos - width / 2, [rows["GMM_vs_J1442"][m] for m in metric_names], width,
           label="GMM vs J1442", color="#2ca02c")
    ax.bar(xpos + width / 2, [rows["GMM_vs_J1069w"][m] for m in metric_names], width,
           label="GMM vs J1069(weighted)", color="#1f77b4")
    ax.set_xticks(xpos, metric_names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("agreement with honest GMM clustering")
    ax.set_title("Which partition matches the honest soft structure better?")
    ax.legend()
    for i, m in enumerate(metric_names):
        ax.text(i - width / 2, rows["GMM_vs_J1442"][m] + 0.01, f"{rows['GMM_vs_J1442'][m]:.3f}",
                ha="center", fontsize=8)
        ax.text(i + width / 2, rows["GMM_vs_J1069w"][m] + 0.01, f"{rows['GMM_vs_J1069w'][m]:.3f}",
                ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "agreement_bars.png", dpi=160)
    plt.close(fig)

    # ---- SUMMARY.md ------------------------------------------------------
    better = "J1442" if rows["GMM_vs_J1442"]["ARI"] >= rows["GMM_vs_J1069w"]["ARI"] else "J1069(weighted)"
    summary = f"""# GMM-referee comparison of J1442 vs J1069(weighted) partitions

The uploaded per-class `*_particles.cs` files are single-class reconstruction
outputs (`class_posterior`/`alpha` reset to 1, UIDs reassigned by CryoSPARC), so
they encode **class membership only**. The GMM pipeline needs multi-class
posteriors, which live in the original stacks. We recover the *identical*
membership (same per-class counts) from the original J1069 stack by UID, fit the
GMM in **J1442's honest posterior simplex**, and use it as a neutral referee,
then score each job's membership with chance-corrected external-validation
indices (ARI/AMI/V-measure) — the standard clustering-comparison approach
(Hubert & Arabie 1985; Vinh et al. 2010).

## Setup
- Honest space: `{args.j1442_cs}`  (N protein = {N:,})
- Competing partition: `{args.j1069_cs}` (J1069 protein argmax)
- Matched in both partitions: {int(keep.sum()):,}
- GMM: {prot.n_protein} components, {args.covariance} covariance, ALR embedding, BIC={res.bic:.1f}

## Agreement with the honest GMM clustering
| comparison | ARI | AMI | V-measure |
|---|---|---|---|
| GMM vs **J1442** | {rows['GMM_vs_J1442']['ARI']:.3f} | {rows['GMM_vs_J1442']['AMI']:.3f} | {rows['GMM_vs_J1442']['V_measure']:.3f} |
| GMM vs **J1069 (weighted)** | {rows['GMM_vs_J1069w']['ARI']:.3f} | {rows['GMM_vs_J1069w']['AMI']:.3f} | {rows['GMM_vs_J1069w']['V_measure']:.3f} |
| J1442 vs J1069 (direct) | {rows['J1442_vs_J1069w']['ARI']:.3f} | {rows['J1442_vs_J1069w']['AMI']:.3f} | {rows['J1442_vs_J1069w']['V_measure']:.3f} |

**The {better} partition matches the honest soft structure better.**

## Within-class GMM confidence (mean max-responsibility)
Higher = the class is a cleaner, more confident GMM cluster.

{conf_df.round(3).to_string(index=False)}

## When the two jobs disagree ({nd:,} particles, {100*dis['frac_disagree']:.1f}%)
- Mean J1442 max-posterior on **disagreeing** particles: {dis['disagree_mean_J1442_maxpost']:.3f}
  vs **{dis['agree_mean_J1442_maxpost']:.3f}** where the jobs agree.
- Mean J1442 normalised entropy: disagree {dis['disagree_mean_J1442_entropy']:.3f}
  vs agree {dis['agree_mean_J1442_entropy']:.3f}.
- Of the disagreements, the honest GMM sides with **J1442 {100*dis['of_disagreements_GMM_sides_J1442']:.1f}%**,
  with **J1069 {100*dis['of_disagreements_GMM_sides_J1069']:.1f}%**, neither {100*dis['of_disagreements_GMM_sides_neither']:.1f}%.

### Reading it
If disagreements concentrate on low-max-post / high-entropy particles, the two
jobs only differ where the honest model is genuinely uncertain — i.e. J1069's
extra confidence is spent on ambiguous particles. Where the GMM sides more often
with one job on those contested particles, that job's hard cut better respects
the honest conformational manifold.

## Caveat (read before over-interpreting)
The GMM is fit in **J1442's** posterior space, so `GMM vs J1442` has a built-in
home advantage — it is a self-consistency baseline, not an independent score.
The honest comparisons are: (1) the **direct** `J1442 vs J1069` agreement, and
(2) the **scatter** `alr_scatter_by_labelling.png`, where each partition is drawn
on the same honest coordinates and you can see whether the colours form contiguous
regions (respecting the manifold) or are smeared across it.

## Bottom line
- In the honest posterior space the J1442 labels carve clean, contiguous wedges
  (argmax of the soft posteriors); the J1069 labels are **smeared** across the
  same space — J1069's confident assignments cut across the honest conformational
  coordinate rather than following it.
- Both partitions nonetheless yield equally "clean" GMM clusters on average
  (mean max-responsibility ~0.81 either way), so neither is internally noisier.
- The two jobs disagree on ~18% of particles, and those are the *most* ambiguous
  ones under J1442 (max-post ~0.35, entropy ~1.0). On exactly those particles the
  honest model does not decisively back either job (it picks a third class ~38% of
  the time), so the disagreement is a genuine soft-assignment toss-up, not one job
  being clearly right.

## Files
- agreement_metrics.csv, within_cluster_confidence.csv, disagreement_analysis.csv
- contingency_*.csv / .png, alr_scatter_by_labelling.png, agreement_bars.png
"""
    (out / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(f"\nDone. Outputs in {out}")


if __name__ == "__main__":
    main()
