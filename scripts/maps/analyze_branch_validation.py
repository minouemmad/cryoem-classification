"""Comprehensive validation of one classification branch's CryoSPARC outputs.

This analyses the downstream CryoSPARC jobs run on a single classification
branch (here: J1069 hard assignments weighted by J1442 posteriors) and asks the
question the cryo-EM / clustering literature actually uses to decide whether a
classification is *real*:

PART A — Image-level re-classification stability (the decisive test)
-------------------------------------------------------------------
We seed CryoSPARC with the branch's per-class maps and let it RE-ASSIGN every
particle from the raw images (Heterogeneous Refinement, J3571) and independently
RE-DERIVE a partition with no alignment (3D Classification, J3579). If the
original classification captured real structure, particles should return to
their original class. We quantify agreement with chance-corrected indices:

    * Adjusted Rand Index (Hubert & Arabie 1985)
    * Adjusted / Normalised Mutual Information (Vinh et al. 2010)
    * Homogeneity / Completeness / V-measure (Rosenberg & Hirschberg 2007)
    * Row-normalised confusion (retention) matrices, Hungarian-aligned.

We also report the NEW per-particle confidence (``class_posterior``) the image
data assigns — the honest measure of how separable the classes are once you go
back to the images, independent of the posterior the classification started
from.

PART B — GMM referee in the honest J1442 simplex
------------------------------------------------
An unsupervised GMM fit in J1442's honest posterior simplex acts as a neutral
reference clustering (external cluster validation). We measure how well each
labelling (input branch assignment, hetero re-assignment, 3D-class re-
derivation, and J1442's own argmax) matches the natural conformational clusters,
plus within-class GMM confidence (cluster cleanliness) and ALR-space silhouette.

Run
---
    python scripts/analyze_branch_validation.py \
        --branch-dir results_J1069/cryosparc_outputs/with_1442_weights \
        --input-index results_J1069/exports_combined/combined_J1069_w1442_class_index.csv \
        --outdir results_J1069/branch_validation_w1442
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
    silhouette_score,
)
from sklearn.metrics.cluster import contingency_matrix

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gmm_pipeline import alr_transform, fit_gmm, load_posteriors

_COLORS = ["#2ca02c", "#d62728", "#1f77b4", "#9467bd", "#ff7f0e", "#8c564b"]
LABELS = ["P6", "P7", "P8"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _align_labels(reference, candidate):
    """Hungarian-permute candidate ids to best match reference ids."""
    C = contingency_matrix(reference, candidate)
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


def _heatmap(M, row_labels, col_labels, title, path, fmt="{:.2f}"):
    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_xlabel("re-assigned class")
    ax.set_ylabel("original class")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                    color="white" if M[i, j] < 0.6 else "black", fontsize=9)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _agreement(a, b):
    h, c, v = homogeneity_completeness_v_measure(a, b)
    return {
        "ARI": adjusted_rand_score(a, b),
        "AMI": adjusted_mutual_info_score(a, b),
        "NMI": normalized_mutual_info_score(a, b),
        "homogeneity": h, "completeness": c, "v_measure": v,
    }


def load_branch_labels(branch_dir: Path):
    """Return dicts uid->new label (int 0/1/2) and uid->class_posterior for the
    hetero-refinement (J3571) and 3D-classification (J3579) jobs."""
    out = {}

    def collect(files):
        labels, conf = {}, {}
        for cls_idx, f in enumerate(files):
            a = np.load(f)
            cp = (a["alignments3D/class_posterior"].astype(float)
                  if "alignments3D/class_posterior" in a.dtype.names
                  else np.full(len(a), np.nan))
            for u, p in zip(a["uid"], cp):
                labels[int(u)] = cls_idx
                conf[int(u)] = float(p)
        return labels, conf

    hetero = branch_dir / "hetero_refinement"
    hetero_files = [
        hetero / "P6" / "cryosparc_P25_J3571_class_00_00102_particles.cs",
        hetero / "P7" / "cryosparc_P25_J3571_class_01_00102_particles.cs",
        hetero / "P8" / "cryosparc_P25_J3571_class_02_00102_particles.cs",
    ]
    out["hetero"] = collect([f for f in hetero_files if f.exists()])

    tdc = branch_dir / "combined" / "3D_classification"
    tdc_files = [
        tdc / "class_0" / "cryosparc_P25_J3579_class_00_00121_particles.cs",
        tdc / "class_1" / "cryosparc_P25_J3579_class_01_00121_particles.cs",
        tdc / "class_2" / "cryosparc_P25_J3579_class_02_00121_particles.cs",
    ]
    out["3dclass"] = collect([f for f in tdc_files if f.exists()])
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--branch-dir", required=True)
    ap.add_argument("--input-index", required=True,
                    help="class_index.csv (uid,class,weight) = branch input assignment")
    ap.add_argument("--j1442-cs", default="data/cryosparc_P25_J1442_00000_particles.cs")
    ap.add_argument("--n-dummies", type=int, default=6)
    ap.add_argument("--outdir", default="results_J1069/branch_validation_w1442")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-points", type=int, default=40000)
    args = ap.parse_args()

    branch_dir = Path(args.branch_dir)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    lab2int = {l: i for i, l in enumerate(LABELS)}

    # ---- input assignment ----
    idx = pd.read_csv(args.input_index)
    idx["lab"] = idx["class"].map(lab2int)
    input_label = {int(u): int(l) for u, l in zip(idx["uid"], idx["lab"])}
    input_weight = {int(u): float(w) for u, w in zip(idx["uid"], idx["weight"])}
    print(f"[input] {len(input_label):,} particles, "
          f"classes {dict(idx['class'].value_counts())}")

    # ---- new labels from the two re-classification jobs ----
    branch = load_branch_labels(branch_dir)

    summary = {"branch_dir": str(branch_dir)}
    agree_rows = []
    conf_rows = []

    for job, (new_label, new_conf) in branch.items():
        common = [u for u in new_label if u in input_label]
        a = np.array([input_label[u] for u in common])      # original
        b = np.array([new_label[u] for u in common])        # re-assigned
        b_al = _align_labels(a, b)
        n = len(common)

        ag = _agreement(a, b_al)
        ag.update({"job": job, "n_matched": n,
                   "n_new_total": len(new_label),
                   "frac_input_recovered": n / len(input_label)})
        agree_rows.append(ag)

        # confusion / retention (row = original, col = re-assigned)
        C = contingency_matrix(a, b_al)
        Cn = _row_norm(C)
        pd.DataFrame(C, index=[f"orig_{l}" for l in LABELS],
                     columns=[f"new_{l}" for l in LABELS]).to_csv(
            out / f"confusion_{job}_counts.csv")
        pd.DataFrame(Cn, index=[f"orig_{l}" for l in LABELS],
                     columns=[f"new_{l}" for l in LABELS]).to_csv(
            out / f"confusion_{job}_rownorm.csv")
        _heatmap(Cn, [f"orig {l}" for l in LABELS], [f"new {l}" for l in LABELS],
             f"{job}: re-assignment retention", out / f"confusion_{job}.png")

        # new image-level confidence
        conf = np.array([new_conf[u] for u in common])
        conf = conf[np.isfinite(conf)]
        conf_rows.append({
            "job": job, "n": len(conf),
            "mean_class_posterior": float(np.mean(conf)),
            "median_class_posterior": float(np.median(conf)),
            "frac>0.5": float(np.mean(conf > 0.5)),
            "frac>0.9": float(np.mean(conf > 0.9)),
            "frac>0.99": float(np.mean(conf > 0.99)),
        })
        print(f"[{job}] matched {n:,}  ARI={ag['ARI']:.3f}  AMI={ag['AMI']:.3f}  "
              f"mean new conf={np.mean(conf):.3f}  frac>0.9={np.mean(conf>0.9):.3f}")

    pd.DataFrame(agree_rows).to_csv(out / "reclassification_agreement.csv", index=False)
    pd.DataFrame(conf_rows).to_csv(out / "new_confidence_stats.csv", index=False)

    # ---- PART B: GMM referee in honest J1442 simplex ----
    print("[referee] loading J1442 honest posteriors")
    prot = load_posteriors(args.j1442_cs, n_dummies=args.n_dummies).protein_only()
    ref_uid = prot.uid.astype(np.uint64)        # keep native uint64 (uids > 2**63)
    X = alr_transform(prot.posterior)
    print(f"[referee] fitting GMM K={len(LABELS)} in ALR simplex "
          f"({len(ref_uid):,} particles)")
    gmm = fit_gmm(X, n_components=len(LABELS),
                  init_hard=prot.hard_class, random_state=args.seed)
    resp = gmm.responsibilities
    gmm_argmax = resp.argmax(axis=1)
    gmm_maxresp = resp.max(axis=1)

    uid_to_row = {int(u): i for i, u in enumerate(ref_uid)}

    labellings = {"input": input_label,
                  "hetero": branch["hetero"][0],
                  "3dclass": branch["3dclass"][0],
                  "j1442_argmax": {int(u): int(c) for u, c in
                                   zip(ref_uid, prot.hard_class)}}

    ref_rows = []
    # subsample once for silhouette (O(n^2)); reused across labellings
    rng = np.random.default_rng(args.seed)
    sil_pool = rng.choice(len(ref_uid), size=min(8000, len(ref_uid)), replace=False)
    sil_set = set(int(s) for s in sil_pool)

    for name, lab in labellings.items():
        rows = [(uid_to_row[u], v) for u, v in lab.items() if u in uid_to_row]
        if not rows:
            continue
        r_idx = np.array([r for r, _ in rows])
        l_val = np.array([v for _, v in rows])
        l_al = _align_labels(gmm_argmax[r_idx], l_val)

        ag = _agreement(gmm_argmax[r_idx], l_al)
        # within-class GMM confidence (cluster cleanliness)
        clean = float(np.mean(gmm_maxresp[r_idx]))
        # silhouette of this labelling in ALR space, on the shared subsample
        row_to_lab = {int(r): int(v) for r, v in zip(r_idx, l_al)}
        keep = np.array([r for r in r_idx if int(r) in sil_set])
        if len(keep) > 50:
            sub_lab = np.array([row_to_lab[int(r)] for r in keep])
            if len(np.unique(sub_lab)) > 1:
                sil = float(silhouette_score(X[keep], sub_lab))
            else:
                sil = np.nan
        else:
            sil = np.nan
        ref_rows.append({"labelling": name, "n": len(r_idx),
                         "ARI_vs_GMM": ag["ARI"], "AMI_vs_GMM": ag["AMI"],
                         "within_class_GMM_maxresp": clean,
                         "alr_silhouette": sil})
        print(f"[referee] {name:12s} n={len(r_idx):,}  ARI_vs_GMM={ag['ARI']:.3f}  "
              f"clean={clean:.3f}  sil={sil:.3f}")

    pd.DataFrame(ref_rows).to_csv(out / "gmm_referee_metrics.csv", index=False)

    gdiag = {
        "gmm_weights": gmm.model.weights_.tolist(),
        "gmm_bic": gmm.bic,
        "gmm_mean_maxresp": float(np.mean(gmm_maxresp)),
        "gmm_frac_maxresp>0.9": float(np.mean(gmm_maxresp > 0.9)),
        "j1442_mean_maxpost": float(np.mean(prot.posterior.max(axis=1))),
    }
    summary["gmm"] = gdiag
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nDone. Outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
