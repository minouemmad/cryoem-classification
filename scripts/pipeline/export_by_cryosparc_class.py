"""Export low-uncertainty per-conformation particle sets grouped by the
CryoSPARC class assignment (argmax of the protein-only posterior).

For overconfident datasets like J1069 the K=3 GMM repartitions the simplex,
so the pipeline's GMM-component grouping does not correspond to CryoSPARC
conformations. This script instead groups by CryoSPARC argmax, which is the
meaningful per-conformation split, and applies a confidence cut on the
CryoSPARC max posterior.

Usage
-----
    python export_by_cryosparc_class.py --cs cryosparc_P25_J1069_00042_particles.cs \
        --n-dummies 6 --resp-threshold 0.9 --outdir results_J1069
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gmm_pipeline import load_posteriors


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cs", required=True, help="CryoSPARC *_particles.cs file")
    p.add_argument("--passthrough-cs", default=None,
                   help="Optional matching passthrough .cs for export")
    p.add_argument("--n-dummies", type=int, default=6)
    p.add_argument("--protein-idx", type=int, nargs="*", default=None)
    p.add_argument("--resp-threshold", type=float, default=0.9,
                   help="CryoSPARC max-posterior confidence cut (default 0.9)")
    p.add_argument("--outdir", default="results")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.outdir) / "exports"
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Loading {args.cs}")
    post = load_posteriors(args.cs, protein_idx=args.protein_idx,
                           n_dummies=args.n_dummies)
    prot = post.protein_only()
    labels = [f"P{int(c)}" for c in post.protein_idx]
    K = prot.n_protein
    print(f"      N_protein={len(prot.uid):,}  K_protein={K}  labels={labels}")

    max_post = prot.posterior.max(axis=1)
    cs_class = prot.hard_class  # 0-based index within protein classes (CryoSPARC argmax)

    lo_mask = max_post > args.resp_threshold
    print(f"[2/3] Confidence cut: CryoSPARC max posterior > {args.resp_threshold}")
    print(f"      {lo_mask.sum():,} / {len(max_post):,} particles pass "
          f"({100 * lo_mask.mean():.1f} %)")
    for k in range(K):
        n_k = int(((cs_class == k) & lo_mask).sum())
        n_all = int((cs_class == k).sum())
        print(f"        CryoSPARC class {labels[k]}: {n_k:,} pass  (of {n_all:,} total)")

    print("[3/3] Writing per-class .cs (grouped by CryoSPARC class)")
    cs_orig = np.load(args.cs)
    uid_to_row = {int(u): i for i, u in enumerate(cs_orig["uid"])}

    passthrough_orig = None
    passthrough_uid_to_row = None
    if args.passthrough_cs:
        passthrough_orig = np.load(args.passthrough_cs)
        if "uid" in passthrough_orig.dtype.names:
            passthrough_uid_to_row = {int(u): i for i, u in enumerate(passthrough_orig["uid"])}

    records = []
    for k in range(K):
        sel = (cs_class == k) & lo_mask
        sel_uids = prot.uid[sel]
        rows = np.array([uid_to_row[int(u)] for u in sel_uids if int(u) in uid_to_row])
        if len(rows) == 0:
            print(f"        {labels[k]}: no matching UIDs — skipping")
            continue
        fname = out / f"low_uncertainty_cryosparc_{labels[k]}.cs"
        with open(fname, "wb") as fh:
            np.save(fh, cs_orig[rows])
        print(f"        {labels[k]}: saved {len(rows):,} particles -> {fname.name}")

        if passthrough_orig is not None:
            if passthrough_uid_to_row is not None:
                pass_rows = np.array([passthrough_uid_to_row[int(u)]
                                      for u in sel_uids if int(u) in passthrough_uid_to_row])
            else:
                pass_rows = rows
            if len(pass_rows):
                pname = out / f"low_uncertainty_cryosparc_{labels[k]}_passthrough.cs"
                with open(pname, "wb") as fh:
                    np.save(fh, passthrough_orig[pass_rows])
                print(f"        {labels[k]} passthrough: saved {len(pass_rows):,} rows -> {pname.name}")

        for u, mp in zip(sel_uids, max_post[sel]):
            records.append({"uid": int(u), "cryosparc_class": labels[k],
                            "max_posterior": float(mp)})

    pd.DataFrame.from_records(records).to_csv(
        out / "low_uncertainty_cryosparc_particles.csv", index=False)
    print(f"      particle list -> low_uncertainty_cryosparc_particles.csv")
    print("done.")


if __name__ == "__main__":
    main()
