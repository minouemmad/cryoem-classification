#!/usr/bin/env python
"""Compare CryoSPARC-class SEPARATION in the latent space of two cryoDRGN runs
across training epochs.

Motivating question (J264 = E1371Q-6SS/ATP): does the *purified* run (junk-
filtered, --ind ind_keep) actually separate the 9 CryoSPARC classes better than
the *fullset* run (all particles), or is purification not buying anything?

Method
------
Each run's z rows are aligned to the CryoSPARC protein-class hard labels by uid
(fullset via the full passthrough; purified via the kept-passthrough .npy that
was built as full_passthrough[sorted(ind_keep)]). To make the comparison fair we
evaluate BOTH runs on the SAME common particle subset (the intersection of uids,
= the purified set), so any difference reflects the *model*, not the particle
count. Per run per epoch we report supervised + geometric separation metrics:

* ``lda_balacc``   5-fold balanced accuracy of an LDA classifier z -> class
                   (the headline supervised-separability number);
* ``min_sep_sd``   minimum pairwise class separation in SD units (project on the
                   centroid-connecting axis; <~2 SD = overlapping / not discrete);
* ``silhouette``   silhouette of the class labelling (subsampled);
* ``knn_selfcons`` mean fraction of a point's k nearest neighbours sharing its
                   class (subsampled) - local purity.

Higher = more separated. The curves vs epoch answer "is one meaningfully better
after some epoch?".

Run with the cryoDRGN env from the repo root, e.g.::

    python scripts/cryodrgn/cryodrgn_run_separation_compare.py \
      --run "fullset:results_cryodrgn/J264/fullset_D256_z10_ep50:data/J264/cryosparc_P7_J264_passthrough_blob_particles_all_classes.cs:data/J264/cryosparc_P7_J264_alignments3D_multi_particles_all_classes.cs" \
      --run "purified:results_cryodrgn/J264/purified_D256_z10_ep75:results_cryodrgn/conformational_landscape/J264_9class_D256_ep50/_passthrough_kept.npy:data/J264/cryosparc_P7_J264_alignments3D_multi_particles_all_classes.cs" \
      --protein-idx 6,7,8,9,10,11,12,13,14 --n-dummies 6 \
      --class-names SC,AC,AO,SEPD,AEPD,V-shaped,NBD1-less,NBD2-less,NBD1-less-wide \
      --stride 3 --subsample 15000 \
      -o results_cryodrgn/J264/run_separation_compare
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
_REPO = os.path.dirname(_SCRIPTS)
for p in (_REPO, _SCRIPTS, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import cryodrgn_latent_gmm as clg


# --------------------------------------------------------------------------- #
def epoch_of(path):
    m = re.search(r"z\.(\d+)\.pkl$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def build_map(passthrough, cs, n_dummies, protein_idx):
    """Return (keep_z_indices, labels, uids) aligning z rows -> CryoSPARC class.

    Computed ONCE per run (independent of epoch): z row order follows the
    passthrough; we map each to its protein-only hard class by uid.
    """
    uid_pass = clg.cs_uids(passthrough)
    prot = clg.load_posteriors(cs, protein_idx=protein_idx,
                               n_dummies=n_dummies).protein_only()
    uid_prot = np.asarray(prot.uid).astype(np.uint64)
    row_of = {int(u): i for i, u in enumerate(uid_prot.tolist())}
    keep_z, keep_rows = [], []
    for i, u in enumerate(uid_pass.tolist()):
        r = row_of.get(int(u))
        if r is not None:
            keep_z.append(i)
            keep_rows.append(r)
    keep_z = np.asarray(keep_z, dtype=np.int64)
    keep_rows = np.asarray(keep_rows, dtype=np.int64)
    labels = np.asarray(prot.hard_class)[keep_rows].astype(int)
    uids = uid_pass[keep_z].astype(np.uint64)
    print(f"[map] {passthrough.split(os.sep)[-1]}: matched {len(keep_z):,} z rows "
          f"to {len(np.unique(labels))} classes")
    return keep_z, labels, uids


def parse_loss(run_log):
    """epoch -> total loss from a cryoDRGN run.log (best-effort)."""
    out = {}
    if not os.path.exists(run_log):
        return out
    with open(run_log, errors="ignore") as fh:
        for line in fh:
            if "loss" not in line.lower():
                continue
            me = re.search(r"Epoch:?\s*\[?(\d+)", line)
            ml = re.search(r"total\s*loss\s*[=:]\s*([0-9.]+)", line, re.I) or \
                re.search(r"\bloss\s*[=:]\s*([0-9.]+)", line, re.I)
            if me and ml:
                out[int(me.group(1))] = float(ml.group(1))
    return out


# --------------------------------------------------------------------------- #
def min_sep_sd(Zs, y):
    """Min (and mean) pairwise class separation in SD along centroid axes."""
    classes = np.unique(y)
    seps = []
    for a in range(len(classes)):
        for b in range(a + 1, len(classes)):
            za = Zs[y == classes[a]]
            zb = Zs[y == classes[b]]
            d = za.mean(0) - zb.mean(0)
            nrm = np.linalg.norm(d)
            if nrm < 1e-9:
                seps.append(0.0)
                continue
            u = d / nrm
            pa = za @ u
            pb = zb @ u
            pooled = np.sqrt(0.5 * (pa.var() + pb.var())) + 1e-9
            seps.append(abs(pa.mean() - pb.mean()) / pooled)
    seps = np.asarray(seps)
    return float(seps.min()), float(seps.mean())


def lda_balacc(Zs, y, seed):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(Zs, y):
        lda = LinearDiscriminantAnalysis()
        lda.fit(Zs[tr], y[tr])
        accs.append(balanced_accuracy_score(y[te], lda.predict(Zs[te])))
    return float(np.mean(accs))


def knn_selfcons(Zs, y, k):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Zs)
    _, idx = nn.kneighbors(Zs)
    same = (y[idx[:, 1:]] == y[:, None]).mean()
    return float(same)


def metrics(Z, y, subsample, seed):
    Zs = StandardScaler().fit_transform(Z)
    ms, mean_sep = min_sep_sd(Zs, y)
    lba = lda_balacc(Zs, y, seed)
    # subsample for the O(n^2)/neighbour metrics
    rng = np.random.default_rng(seed)
    n = len(Zs)
    if subsample and n > subsample:
        sel = rng.choice(n, subsample, replace=False)
    else:
        sel = np.arange(n)
    Zsub, ysub = Zs[sel], y[sel]
    try:
        sil = float(silhouette_score(Zsub, ysub))
    except Exception:
        sil = float("nan")
    knn = knn_selfcons(Zsub, ysub, k=15)
    return {"lda_balacc": lba, "min_sep_sd": ms, "mean_sep_sd": mean_sep,
            "silhouette": sil, "knn_selfcons": knn}


# --------------------------------------------------------------------------- #
def analyse(args):
    os.makedirs(args.out, exist_ok=True)
    protein_idx = [int(x) for x in args.protein_idx.split(",")]

    runs = []
    for spec in args.run:
        label, zdir, passthrough, cs = spec.split(":", 3)
        keep_z, labels, uids = build_map(passthrough, cs, args.n_dummies, protein_idx)
        zpaths = sorted(glob.glob(os.path.join(zdir, "z.*.pkl")), key=epoch_of)
        zpaths = [p for p in zpaths if epoch_of(p) >= 0]
        runs.append({"label": label, "zdir": zdir, "keep_z": keep_z,
                     "labels": labels, "uids": uids, "zpaths": zpaths,
                     "loss": parse_loss(os.path.join(zdir, "run.log"))})

    # common particle subset (intersection of uids across runs)
    common = None
    for r in runs:
        s = set(r["uids"].tolist())
        common = s if common is None else (common & s)
    common = np.array(sorted(common), dtype=np.uint64)
    print(f"[common] {len(common):,} particles shared across "
          f"{len(runs)} runs")

    # per-run row index for the common uids (+ their labels)
    for r in runs:
        pos = {int(u): i for i, u in enumerate(r["uids"].tolist())}
        r["common_rows"] = np.array([pos[int(u)] for u in common.tolist()],
                                    dtype=np.int64)
        r["common_labels"] = r["labels"][r["common_rows"]]

    # epoch sweep
    results = {r["label"]: [] for r in runs}
    for r in runs:
        eps = [epoch_of(p) for p in r["zpaths"]]
        sel = eps[:: max(1, args.stride)]
        if eps and eps[-1] not in sel:
            sel.append(eps[-1])
        for p in r["zpaths"]:
            e = epoch_of(p)
            if e not in sel:
                continue
            z = clg.load_latent(p)
            z_a = z[r["keep_z"]]                 # aligned to labels/uids order
            Z = z_a[r["common_rows"]]            # restrict to common particles
            m = metrics(Z, r["common_labels"], args.subsample, args.seed)
            m["epoch"] = e
            m["loss"] = r["loss"].get(e)
            results[r["label"]].append(m)
            print(f"[{r['label']}] ep{e:>3}  LDA={m['lda_balacc']:.3f}  "
                  f"minSep={m['min_sep_sd']:.2f}SD  sil={m['silhouette']:.3f}  "
                  f"kNN={m['knn_selfcons']:.3f}")

    _plot(args, runs, results, common)
    _write(args, runs, results, common, protein_idx)
    _verdict(results)


def _plot(args, runs, results, common):
    panels = [("lda_balacc", "LDA 5-fold balanced acc (supervised separability)"),
              ("min_sep_sd", "min pairwise class sep (SD)  [>2 = discrete]"),
              ("silhouette", "silhouette of class labels"),
              ("knn_selfcons", "kNN self-consistency (local purity)")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (key, title) in zip(axes.ravel(), panels):
        for r in runs:
            rows = sorted(results[r["label"]], key=lambda d: d["epoch"])
            xs = [d["epoch"] for d in rows]
            ys = [d[key] for d in rows]
            ax.plot(xs, ys, "o-", ms=4, label=r["label"])
        ax.set_xlabel("epoch")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle(f"J264 latent class-separation vs epoch  "
                 f"(common {len(common):,} particles, 9 CryoSPARC classes)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(args.out, "separation_vs_epoch.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


def _write(args, runs, results, common, protein_idx):
    import csv
    keys = ["run", "epoch", "lda_balacc", "min_sep_sd", "mean_sep_sd",
            "silhouette", "knn_selfcons", "loss"]
    with open(os.path.join(args.out, "separation_vs_epoch.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in runs:
            for d in sorted(results[r["label"]], key=lambda d: d["epoch"]):
                row = {"run": r["label"]}
                row.update({k: d.get(k) for k in keys if k != "run"})
                w.writerow(row)
    with open(os.path.join(args.out, "separation_vs_epoch.json"), "w") as fh:
        json.dump({"common_particles": int(len(common)),
                   "protein_idx": protein_idx, "results": results}, fh, indent=2)
    print(f"[write] {args.out}")


def _verdict(results):
    print("\n===== SUMMARY (last available epoch per run) =====")
    last = {}
    for label, rows in results.items():
        if not rows:
            continue
        d = sorted(rows, key=lambda x: x["epoch"])[-1]
        last[label] = d
        print(f"  {label:<10} ep{d['epoch']:>3}: LDA={d['lda_balacc']:.3f}  "
              f"minSep={d['min_sep_sd']:.2f}SD  sil={d['silhouette']:.3f}  "
              f"kNN={d['knn_selfcons']:.3f}")
    if len(last) == 2:
        (la, da), (lb, db) = list(last.items())
        dlda = da["lda_balacc"] - db["lda_balacc"]
        better = la if dlda > 0 else lb
        if abs(dlda) <= 0.02:
            msg = "no meaningful difference between runs"
        else:
            msg = f"{better} separates the classes meaningfully better"
        print(f"\n  ΔLDA ({la} - {lb}) = {dlda:+.3f}  ({msg})")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True,
                    help='"LABEL:ZDIR:PASSTHROUGH:CS" (repeatable)')
    ap.add_argument("--protein-idx", default="6,7,8,9,10,11,12,13,14")
    ap.add_argument("--n-dummies", type=int, default=6)
    ap.add_argument("--class-names", default="")
    ap.add_argument("--stride", type=int, default=3, help="use every Nth epoch")
    ap.add_argument("--subsample", type=int, default=15000,
                    help="subsample for silhouette/kNN (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--out",
                    default="results_cryodrgn/J264/run_separation_compare")
    return ap


def main(argv=None):
    analyse(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
