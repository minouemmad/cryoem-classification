#!/usr/bin/env python
"""Re-plot the cryoDRGN state-confusion figure with clear, method-explicit axes.

The old ``confusion.png`` labelled its axes "true state" / "observed state",
which is misleading: neither axis is a ground truth. This script produces two
row-normalised matrices with unambiguous labels and biological class names.

1. ``confusion.png`` -- CryoSPARC vs cryoDRGN (cross-method agreement).
     Row = CryoSPARC hetero-refine class (argmax of ``class_posterior``).
     Col = cryoDRGN latent class (argmax of the latent soft-assignment).
     Entry[i, j] = fraction of CryoSPARC class *i* that cryoDRGN assigns to *j*.
     A clean block-diagonal = the two methods agree. Needs ``--cs``.

2. ``confusion_selfconsistency.png`` -- within cryoDRGN only.
     Row = soft membership weight to a state (cryoDRGN latent QDA).
     Col = hard assignment (argmax, same cryoDRGN latent).
     Diagonal = how self-consistent each cryoDRGN state is (soft vs hard).
     Uses only ``soft_probabilities.csv``.

Both assignments are *observed* (from different methods); neither is "true".

Run with the cryoDRGN env from repo root::

    python scripts/cryodrgn/plot_state_confusion.py \
      --soft-csv results_cryodrgn/J1442_gP25_WT_POSE_BIAS/confidence_5class/J1497_5class_soft_probabilities.csv \
      --cs data/cryosparc_P25_J1497_00000_particles.cs \
      --protein-idx 6 7 8 9 10 --n-dummies 6 \
      -o results_cryodrgn/J1442_gP25_WT_POSE_BIAS/confidence_5class
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))   # scripts/cryodrgn
_SCRIPTS = os.path.dirname(_HERE)                     # scripts
_REPO = os.path.dirname(_SCRIPTS)                     # repo root
for _p in (_REPO, _SCRIPTS, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import class_names as cnames
from gmm_pipeline.confusion import soft_posterior_confusion
from gmm_pipeline.data_io import load_posteriors


def read_soft_csv(path):
    """Return (uids:list[int], states:list[int], P:(N,K), hard_idx:(N,))."""
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        pcols = [c for c in header if c.startswith("p_C")]
        states = [int(c[3:]) for c in pcols]
        col = {c: header.index(c) for c in header}
        pidx = [col[c] for c in pcols]
        hcol = col.get("hard_state")
        uids, rows, hard = [], [], []
        state_to_k = {s: k for k, s in enumerate(states)}
        for r in reader:
            if not r:
                continue
            uids.append(int(r[col["uid"]]))
            rows.append([float(r[i]) for i in pidx])
            hard.append(state_to_k[int(r[hcol][1:])] if hcol is not None else -1)
    P = np.asarray(rows, float)
    return uids, states, P, np.asarray(hard, int)


def plot_confusion(C, row_labels, col_labels, title, xlabel, ylabel, out):
    K = len(row_labels)
    fig, ax = plt.subplots(figsize=(1.6 * K + 3.5, 1.4 * K + 2.5))
    im = ax.imshow(C, cmap="magma", vmin=0, vmax=1, aspect="equal")
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center",
                    color="white" if C[i, j] < 0.6 else "black", fontsize=10)
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="row-normalised fraction")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--soft-csv", required=True,
                    help="cryoDRGN soft_probabilities.csv (uid, p_C*, hard_state)")
    ap.add_argument("--cs", default=None,
                    help="CryoSPARC particles .cs for the cross-method matrix")
    ap.add_argument("--protein-idx", type=int, nargs="+", default=None)
    ap.add_argument("--n-dummies", type=int, default=6)
    ap.add_argument("--dataset", default=None,
                    help="J1442/J1497/J264/J325 (auto-detected from paths)")
    ap.add_argument("-o", "--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    dset = args.dataset or cnames.guess_dataset(args.cs, args.soft_csv, args.outdir)
    uids, states, P, hard = read_soft_csv(args.soft_csv)
    labels = [cnames.label(dset, s) for s in states]
    print(f"[load] {len(uids):,} particles, {len(states)} states {states} "
          f"(dataset={dset})")

    # 1) within-cryoDRGN self-consistency (soft membership vs hard argmax)
    C_self = soft_posterior_confusion(P)
    plot_confusion(
        C_self, labels, labels,
        title=f"cryoDRGN latent state self-consistency ({dset})\n"
              f"soft membership vs its own hard assignment",
        xlabel="cryoDRGN hard assignment (argmax)",
        ylabel="cryoDRGN soft membership",
        out=os.path.join(args.outdir, "confusion_selfconsistency.png"))

    # 2) cross-method CryoSPARC vs cryoDRGN (the headline confusion.png)
    if args.cs:
        prot = load_posteriors(args.cs, args.protein_idx, args.n_dummies).protein_only()
        cs_uid_to_class = {int(u): int(c) for u, c in zip(prot.uid, prot.hard_class)}
        drgn_uid_to_class = dict(zip(uids, hard))
        common = [u for u in uids if u in cs_uid_to_class]
        K = len(states)
        M = np.zeros((K, K))
        for u in common:
            M[cs_uid_to_class[u], drgn_uid_to_class[u]] += 1
        rowsum = M.sum(1, keepdims=True)
        C_cross = np.divide(M, rowsum, out=np.zeros_like(M), where=rowsum > 0)
        print(f"[cross] matched {len(common):,} uids to CryoSPARC labels; "
              f"agreement (trace/N) = {np.trace(M) / max(len(common), 1):.3f}")
        plot_confusion(
            C_cross, labels, labels,
            title=f"CryoSPARC vs cryoDRGN class agreement ({dset})\n"
                  f"both are observed assignments (different methods), not ground truth",
            xlabel="cryoDRGN latent class (argmax)",
            ylabel="CryoSPARC hetero-refine class (argmax)",
            out=os.path.join(args.outdir, "confusion.png"))
    else:
        print("[skip] no --cs given; wrote only the self-consistency matrix. "
              "Pass --cs for the CryoSPARC-vs-cryoDRGN confusion.png")


if __name__ == "__main__":
    main()
