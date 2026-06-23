#!/usr/bin/env python
"""Export per-class particle subsets from cryoDRGN latent-GMM assignments.

Reads per_particle.npz (written by cryodrgn_latent_gmm.py) and exports two
kinds of CryoSPARC-compatible .cs particle files for each class:

  cryodrgn_class_<NAME>.cs     -- ALL particles hard-assigned to that class
  cryodrgn_class_<NAME>_hc.cs  -- HIGH-CONFIDENCE subset
                                   (CryoSPARC hard class == cryoDRGN hard class
                                    AND JS divergence < --js-pct percentile)

The .cs output files are numpy structured arrays (same format as CryoSPARC
output .cs files) subsetted from the passthrough .cs.  Import them into
CryoSPARC via:  Jobs > Import Particles > "Import from external .cs file".
Then run ab-initio reconstruction + NU-refinement on each subset.

Run from repo root with the cryodrgn-py310 env::

    python scripts/cryodrgn/export_cryodrgn_subsets.py \\
      --npz  results_cryodrgn/J1442_real/latent_gmm_z10/per_particle.npz \\
      --passthrough-cs  data/cryosparc_P25_J1442_passthrough_particles_all_classes.cs \\
      --protein-idx 6 7 8 \\
      --js-pct 33 \\
      -o  results_cryodrgn/J1442_real/cryodrgn_subsets
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _save_cs(path: str, arr: np.ndarray) -> None:
    """Save a numpy structured array as a CryoSPARC .cs file (no .npy suffix)."""
    with open(path, "wb") as fh:
        np.save(fh, arr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True,
                    help="per_particle.npz from cryodrgn_latent_gmm.py")
    ap.add_argument("--passthrough-cs", required=True,
                    help="passthrough .cs (has blob/pose/CTF for all particles)")
    ap.add_argument("--protein-idx", type=int, nargs="+", default=[6, 7, 8],
                    help="protein class indices used during cryodrgn_latent_gmm.py run")
    ap.add_argument("--js-pct", type=float, default=33.0,
                    help="JS divergence percentile threshold for high-confidence "
                         "subset (default: 33rd pct = bottom third of divergences)")
    ap.add_argument("-o", "--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    class_names = [f"P{j}" for j in args.protein_idx]
    K = len(class_names)

    # ------------------------------------------------------------------ #
    # load per-particle data
    d = np.load(args.npz)
    npz_uids   = d["uid"].astype(np.uint64)
    drgn_post  = d["cryodrgn_gmm_posterior"]   # (N, K) col j = class_names[j]
    js_div     = d["js_divergence"]
    agreement  = d["agreement"]

    hard_class = drgn_post.argmax(axis=1)       # 0=class_names[0], etc.

    # JS threshold for high-confidence
    js_thresh = float(np.percentile(js_div, args.js_pct))
    print(f"[export] JS threshold ({args.js_pct:.0f}th pct): {js_thresh:.4f}  "
          f"(median={float(np.median(js_div)):.4f})")
    print(f"[export] total particles: {len(npz_uids):,}  |  "
          f"agree: {int(agreement.sum()):,}  ({100*agreement.mean():.1f}%)")

    # ------------------------------------------------------------------ #
    # load passthrough .cs and build uid -> row index map
    cs = np.load(args.passthrough_cs)
    cs_uids = cs["uid"].astype(np.uint64)
    uid_to_row = {int(u): i for i, u in enumerate(cs_uids.tolist())}
    print(f"[export] passthrough .cs: {len(cs_uids):,} rows")

    # ------------------------------------------------------------------ #
    # export per class
    print(f"\n{'class':<8} {'full':>8} {'hc':>8}  "
          f"({'agree&JS<{:.2f}'.format(js_thresh)})")
    print("-" * 48)

    rows_summary = []
    for k, name in enumerate(class_names):
        mask_full = (hard_class == k)

        # high-confidence: CryoSPARC hard class matches cryoDRGN hard class
        # AND JS < threshold (soft posteriors are similar, not just hard labels)
        mask_hc = mask_full & (agreement == 1) & (js_div < js_thresh)

        for label, mask in [("full", mask_full), ("hc", mask_hc)]:
            sel_uids = npz_uids[mask]
            rows = []
            missing = 0
            for u in sel_uids.tolist():
                r = uid_to_row.get(int(u))
                if r is not None:
                    rows.append(r)
                else:
                    missing += 1
            if missing:
                print(f"  WARNING: {missing} UIDs not found in passthrough .cs")
            rows = np.array(rows, dtype=np.intp)
            subset = cs[rows]

            tag = "" if label == "full" else "_hc"
            out_path = os.path.join(args.outdir, f"cryodrgn_class_{name}{tag}.cs")
            _save_cs(out_path, subset)
            rows_summary.append((name, label, len(rows)))

        n_full = sum(r[2] for r in rows_summary if r[0] == name and r[1] == "full")
        n_hc   = sum(r[2] for r in rows_summary if r[0] == name and r[1] == "hc")
        pct_hc = 100 * n_hc / max(n_full, 1)
        print(f"{name:<8} {n_full:>8,} {n_hc:>8,}  ({pct_hc:.0f}% of class)")

    print("\n[export] wrote .cs files to", args.outdir)
    print("\n--- CryoSPARC import + refinement instructions ---")
    print("For each .cs file:")
    print("  1. CryoSPARC > Import Particles > select the .cs file")
    print("  2. Run Ab-initio Reconstruction (K=1, default settings)")
    print("  3. Use the ab-initio map as input to NU-Refinement")
    print("  Full class  -> gives cryoDRGN-defined class map (compare to CryoSPARC map)")
    print("  HC subset   -> gives higher-confidence map (fewer particles, purer signal)")
    print("\nFiles:")
    for name, label, n in rows_summary:
        tag = "" if label == "full" else "_hc"
        fname = f"cryodrgn_class_{name}{tag}.cs"
        print(f"  {fname:<40}  {n:>8,} particles")


if __name__ == "__main__":
    main()
