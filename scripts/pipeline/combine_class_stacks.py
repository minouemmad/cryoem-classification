"""Combine disjoint per-class CryoSPARC particle exports into ONE full stack.

The per-class hard exports produced by ``export_weighted_by_class.py --assign-cs``
(e.g. ``hardJ1069_w1442_b1_P6.cs`` / ``..._P7.cs`` / ``..._P8.cs``) are *disjoint*
subsets that together cover the full particle stack. Heterogeneous Refinement,
3D Classification and 3D Variability Analysis all need the **whole** stack as a
single input, not three pre-sorted class inputs. This script concatenates the
per-class particle blobs (and their passthroughs) into one combined
``*_particles.cs`` + ``*_passthrough.cs`` pair that can be imported with
"Import Particle Stack".

Weight preservation
-------------------
The per-particle J1442 posterior weight lives in the passthrough scale field
(``alignments3D/alpha`` by default). Concatenation keeps every particle's
``alpha`` exactly, so no weight information is lost. The original hard-class
assignment (which export each particle came from) is written to a sidecar
``*_class_index.csv`` (uid, class, weight) so it can be recovered for
post-hoc confusion / ARI analysis after the new job re-derives a partition.

Usage
-----
    python scripts/combine_class_stacks.py \
        --inputs results_J1069/exports_weighted/hardJ1069_w1442_b1_P6.cs \
                 results_J1069/exports_weighted/hardJ1069_w1442_b1_P7.cs \
                 results_J1069/exports_weighted/hardJ1069_w1442_b1_P8.cs \
        --out-prefix combined_J1069_w1442 \
        --outdir results_J1069/exports_combined

Each ``--inputs`` entry is a particle ``.cs``; its passthrough is assumed to be
the sibling ``<name>_passthrough.cs``. Class labels are parsed from the trailing
``P<n>`` token of each filename (override with ``--labels``).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", required=True,
                   help="Per-class particle .cs files to concatenate (in order).")
    p.add_argument("--out-prefix", required=True,
                   help="Basename for the combined outputs (no extension).")
    p.add_argument("--outdir", required=True, help="Output directory.")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Optional class label per input (default: parse P<n> "
                        "from each filename).")
    p.add_argument("--scale-field", default="alignments3D/alpha",
                   help="Passthrough field holding the per-particle weight "
                        "(default alignments3D/alpha).")
    p.add_argument("--reset-weights", action="store_true",
                   help="Set the scale field to 1.0 in the combined passthrough "
                        "so 3DVA / 3D Classification treat every particle "
                        "equally (unbiased re-derivation). The original weights "
                        "are still recorded in the *_class_index.csv sidecar.")
    return p.parse_args()


def label_from_name(path: Path) -> str:
    m = re.search(r"(P\d+)", path.stem)
    return m.group(1) if m else path.stem


def main():
    args = parse_args()
    inputs = [Path(p) for p in args.inputs]
    labels = args.labels or [label_from_name(p) for p in inputs]
    if len(labels) != len(inputs):
        raise SystemExit("ERROR: number of --labels must match number of --inputs")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    part_chunks, pass_chunks = [], []
    sidecar = []
    for cs_path, lab in zip(inputs, labels):
        pass_path = cs_path.with_name(cs_path.stem + "_passthrough.cs")
        if not cs_path.exists():
            raise SystemExit(f"ERROR: missing particle file {cs_path}")
        if not pass_path.exists():
            raise SystemExit(f"ERROR: missing passthrough file {pass_path}")

        part = np.load(cs_path)
        pas = np.load(pass_path)
        if len(part) != len(pas):
            raise SystemExit(
                f"ERROR: {cs_path.name} particle/passthrough length mismatch "
                f"({len(part)} vs {len(pas)})")
        # Per-file the two blobs are written in the same uid order; verify.
        if not np.array_equal(part["uid"].astype(np.uint64),
                              pas["uid"].astype(np.uint64)):
            raise SystemExit(
                f"ERROR: {cs_path.name} particle/passthrough uid order differs")

        weight = (pas[args.scale_field].astype(np.float64)
                  if args.scale_field in pas.dtype.names else np.full(len(pas), np.nan))
        for u, w in zip(pas["uid"], weight):
            sidecar.append({"uid": int(u), "class": lab, "weight": float(w)})

        part_chunks.append(part)
        pass_chunks.append(pas)
        print(f"  {lab:>4}: {len(part):,} particles  "
              f"(mean weight {np.nanmean(weight):.4f})  <- {cs_path.name}")

    combined_part = np.concatenate(part_chunks)
    combined_pass = np.concatenate(pass_chunks)

    # Sanity: combined particle and passthrough must stay uid-aligned row-for-row.
    if not np.array_equal(combined_part["uid"].astype(np.uint64),
                          combined_pass["uid"].astype(np.uint64)):
        raise SystemExit("ERROR: combined particle/passthrough uid order differs")
    n_unique = len(np.unique(combined_part["uid"].astype(np.uint64)))
    if n_unique != len(combined_part):
        print(f"  WARNING: {len(combined_part) - n_unique:,} duplicate uids "
              f"across inputs (subsets not disjoint).")

    if args.reset_weights and args.scale_field in combined_pass.dtype.names:
        sd = combined_pass.dtype[args.scale_field].base
        combined_pass[args.scale_field] = np.ones(len(combined_pass)).astype(sd)
        print("  reset scale field to 1.0 (weights preserved in class_index.csv)")

    part_out = outdir / f"{args.out_prefix}_particles.cs"
    pass_out = outdir / f"{args.out_prefix}_passthrough.cs"
    with open(part_out, "wb") as fh:
        np.save(fh, combined_part)
    with open(pass_out, "wb") as fh:
        np.save(fh, combined_pass)

    sidecar_df = pd.DataFrame.from_records(sidecar)
    sidecar_out = outdir / f"{args.out_prefix}_class_index.csv"
    sidecar_df.to_csv(sidecar_out, index=False)

    print(f"\nCombined {len(combined_part):,} particles "
          f"({n_unique:,} unique uids) from {len(inputs)} classes.")
    print(f"  particles   -> {part_out}")
    print(f"  passthrough -> {pass_out}")
    print(f"  class index -> {sidecar_out}")
    print("\nImport both .cs files with 'Import Particle Stack', then run "
          "Heterogeneous Refinement / 3D Classification / 3DVA on the result.")


if __name__ == "__main__":
    main()
