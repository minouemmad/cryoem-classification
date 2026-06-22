"""Export class-grouped .cs files across confidence thresholds.

This sweeps confidence cuts (r) and exports per-class particle stacks for each
threshold.

By default, r is the max GMM responsibility from
<outdir>/gmm/responsibilities.npy, matching existing low-uncertainty exports.
Optionally, r can be CryoSPARC max posterior over protein classes.

Example
-------
python scripts/export_threshold_sweep.py \
  --cs data/cryosparc_P25_J1442_00000_particles.cs \
  --passthrough-cs data/cryosparc_P25_J1442_passthrough_particles_all_classes.cs \
  --outdir results_J1442 \
  --thresholds 0.6 0.7 0.8 0.9
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gmm_pipeline import load_posteriors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cs", required=True, help="CryoSPARC *_particles.cs file")
    parser.add_argument(
        "--passthrough-cs",
        default=None,
        help="Optional matching passthrough .cs file to export with the same selections",
    )
    parser.add_argument("--n-dummies", type=int, default=6)
    parser.add_argument("--protein-idx", type=int, nargs="*", default=None)
    parser.add_argument(
        "--metric",
        choices=["gmm", "cryosparc"],
        default="gmm",
        help="Confidence metric for r threshold: 'gmm' max responsibility (default), or 'cryosparc' max posterior",
    )
    parser.add_argument(
        "--responsibilities",
        default=None,
        help="Path to responsibilities.npy (used in metric=gmm). Defaults to <outdir>/gmm/responsibilities.npy",
    )
    parser.add_argument(
        "--label-csv",
        default=None,
        help="Optional CSV with a 'class' column to name components (used in metric=gmm). Defaults to <outdir>/populations/conformational_populations.csv",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.6, 0.7, 0.8, 0.9],
        help="Thresholds r where particles pass if max(metric) > r",
    )
    parser.add_argument("--outdir", default="results")
    parser.add_argument(
        "--subdir",
        default="exports_threshold_sweep",
        help="Subdirectory under outdir where exports are written",
    )
    return parser.parse_args()


def thr_tag(threshold: float) -> str:
    return f"r{threshold:.1f}".replace(".", "p")


def load_gmm_labels(label_csv: Path, n_components: int) -> list[str]:
    if label_csv.exists():
        df = pd.read_csv(label_csv)
        if "class" in df.columns:
            labels = [str(x) for x in df["class"].tolist()]
            if len(labels) == n_components:
                return labels
    return [f"C{i}" for i in range(n_components)]


def main() -> None:
    args = parse_args()

    out_dir = Path(args.outdir) / args.subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading class assignments and confidence metric ({args.metric})")

    if args.metric == "gmm":
        resp_path = Path(args.responsibilities) if args.responsibilities else (Path(args.outdir) / "gmm" / "responsibilities.npy")
        resp = np.load(resp_path)
        max_score = resp.max(axis=1)
        cls_idx = resp.argmax(axis=1)
        labels = load_gmm_labels(
            Path(args.label_csv) if args.label_csv else (Path(args.outdir) / "populations" / "conformational_populations.csv"),
            resp.shape[1],
        )
        print(f"      responsibilities={resp_path}")
        print(f"      N={resp.shape[0]:,}  K={resp.shape[1]}  labels={labels}")
    else:
        post = load_posteriors(args.cs, protein_idx=args.protein_idx, n_dummies=args.n_dummies)
        prot = post.protein_only()
        labels = [f"P{int(c)}" for c in post.protein_idx]
        max_score = prot.posterior.max(axis=1)
        cls_idx = prot.hard_class
        print(f"      N_protein={len(prot.uid):,}  K_protein={prot.n_protein}  labels={labels}")

    print("[2/5] Loading source .cs tables")
    cs_orig = np.load(args.cs)
    uid_to_row = {int(u): i for i, u in enumerate(cs_orig["uid"])}

    if args.metric == "gmm":
        if len(cs_orig) != len(max_score):
            raise ValueError(
                f"Length mismatch: len(cs)={len(cs_orig):,} but len(responsibilities)={len(max_score):,}"
            )
        sel_uids_all = np.asarray(cs_orig["uid"])
    else:
        sel_uids_all = prot.uid

    passthrough_orig = None
    passthrough_uid_to_row = None
    if args.passthrough_cs:
        passthrough_orig = np.load(args.passthrough_cs)
        if "uid" in passthrough_orig.dtype.names:
            passthrough_uid_to_row = {
                int(u): i for i, u in enumerate(passthrough_orig["uid"])
            }

    print("[3/5] Sweeping thresholds and exporting .cs")
    summary_rows = []

    for threshold in sorted(set(args.thresholds)):
        tag = thr_tag(threshold)
        pass_mask = max_score > threshold
        n_pass = int(pass_mask.sum())
        frac = 100.0 * pass_mask.mean()
        print(f"      threshold > {threshold:.1f}: {n_pass:,} / {len(max_score):,} ({frac:.1f}%)")

        for k, label in enumerate(labels):
            sel = (cls_idx == k) & pass_mask
            sel_uids = sel_uids_all[sel]

            rows = np.array([uid_to_row[int(u)] for u in sel_uids if int(u) in uid_to_row])
            n_rows = int(len(rows))

            summary_rows.append(
                {
                    "threshold": float(threshold),
                    "threshold_tag": tag,
                    "class_label": label,
                    "n_particles": n_rows,
                    "n_total_pass": n_pass,
                    "fraction_total_pass": float(pass_mask.mean()),
                }
            )

            if n_rows == 0:
                print(f"        {label}: 0 particles")
                continue

            fname = out_dir / f"low_uncertainty_{tag}_cryosparc_{label}.cs"
            with open(fname, "wb") as fh:
                np.save(fh, cs_orig[rows])
            print(f"        {label}: saved {n_rows:,} -> {fname.name}")

            if passthrough_orig is not None:
                if passthrough_uid_to_row is not None:
                    pass_rows = np.array(
                        [
                            passthrough_uid_to_row[int(u)]
                            for u in sel_uids
                            if int(u) in passthrough_uid_to_row
                        ]
                    )
                else:
                    pass_rows = rows

                if len(pass_rows):
                    pname = out_dir / f"low_uncertainty_{tag}_cryosparc_{label}_passthrough.cs"
                    with open(pname, "wb") as fh:
                        np.save(fh, passthrough_orig[pass_rows])

    print("[4/5] Writing summary CSVs")
    summary_df = pd.DataFrame.from_records(summary_rows)
    summary_df.to_csv(out_dir / "threshold_particle_counts_by_class.csv", index=False)

    totals_df = (
        summary_df[["threshold", "threshold_tag", "n_total_pass", "fraction_total_pass"]]
        .drop_duplicates()
        .sort_values("threshold")
        .rename(columns={"n_total_pass": "n_particles_pass"})
    )
    totals_df.to_csv(out_dir / "threshold_particle_counts_total.csv", index=False)

    print("[5/5] Done")
    print(f"      Wrote exports and summaries to: {out_dir}")


if __name__ == "__main__":
    main()
