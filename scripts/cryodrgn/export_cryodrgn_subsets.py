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
                    help="JS divergence percentile for the STRICT high-confidence "
                         "subset (default: 33rd pct = bottom third of divergences)")
    ap.add_argument("--js-pct-permissive", type=float, default=66.0,
                    help="JS divergence percentile for the PERMISSIVE confidence "
                         "subset (default: 66th pct). John: the strict >=0.9/33pct cut "
                         "removes too many particles; this keeps roughly twice as many.")
    ap.add_argument("--inject-alpha-weight", action="store_true",
                    help="bake the cryoDRGN max responsibility into alignments3D/alpha "
                         "of each exported .cs, so CryoSPARC 'Homogeneous Reconstruction "
                         "Only' weights every particle by its cryoDRGN confidence. "
                         "Without this flag alpha is left untouched (plain hard subset).")
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
    max_resp = drgn_post.max(axis=1)            # cryoDRGN confidence per particle

    # JS thresholds (strict + permissive) for confidence subsets
    js_thresh = float(np.percentile(js_div, args.js_pct))
    js_thresh_perm = float(np.percentile(js_div, args.js_pct_permissive))
    print(f"[export] strict JS threshold ({args.js_pct:.0f}th pct): {js_thresh:.4f}  "
          f"| permissive ({args.js_pct_permissive:.0f}th pct): {js_thresh_perm:.4f}  "
          f"(median={float(np.median(js_div)):.4f})")
    print(f"[export] total particles: {len(npz_uids):,}  |  "
          f"agree: {int(agreement.sum()):,}  ({100*agreement.mean():.1f}%)")

    # ------------------------------------------------------------------ #
    # load passthrough .cs and build uid -> row index map
    cs = np.load(args.passthrough_cs)
    cs_uids = cs["uid"].astype(np.uint64)
    uid_to_row = {int(u): i for i, u in enumerate(cs_uids.tolist())}
    print(f"[export] passthrough .cs: {len(cs_uids):,} rows")

    has_alpha = "alignments3D/alpha" in (cs.dtype.names or ())
    if args.inject_alpha_weight and not has_alpha:
        print("  WARNING: passthrough has no alignments3D/alpha field; "
              "cannot inject weight. Exporting plain subsets instead.")

    # ------------------------------------------------------------------ #
    # sidecar: cryoDRGN posteriors for EVERY particle, keyed by uid. Lets you
    # join the cryoDRGN class probabilities back onto any subset in CryoSPARC
    # (e.g. with a Custom/External job) without re-running anything.
    sidecar = os.path.join(args.outdir, "cryodrgn_posteriors.csv")
    import csv as _csv
    with open(sidecar, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["uid"] + [f"post_{n}" for n in class_names]
                   + ["hard_class", "max_resp", "agreement", "js_divergence"])
        for i in range(len(npz_uids)):
            w.writerow([int(npz_uids[i])]
                       + [f"{drgn_post[i, j]:.6f}" for j in range(K)]
                       + [class_names[hard_class[i]], f"{max_resp[i]:.6f}",
                          int(agreement[i]), f"{js_div[i]:.6f}"])
    print(f"[export] wrote posterior sidecar: {sidecar}")

    # ------------------------------------------------------------------ #
    # export per class. Four confidence tiers (John's meeting notes):
    #   full   = ALL particles hard-assigned to the class (run NU on cryoDRGN particles)
    #   hard   = hard-assignments only: cryoDRGN argmax == CryoSPARC argmax (no JS cut)
    #   hcperm = permissive low-uncertainty: agree AND JS < permissive threshold
    #   hc     = strict low-uncertainty:     agree AND JS < strict threshold
    tag_of = {"full": "", "hard": "_hard", "hcperm": "_hcperm", "hc": "_hc"}
    print(f"\n{'class':<8} {'full':>9} {'hard':>9} {'hcperm':>9} {'hc':>9}")
    print("-" * 50)

    rows_summary = []
    for k, name in enumerate(class_names):
        mask_full = (hard_class == k)
        mask_hard = mask_full & (agreement == 1)
        mask_hcperm = mask_hard & (js_div < js_thresh_perm)
        mask_hc = mask_hard & (js_div < js_thresh)

        counts = {}
        for label, mask in [("full", mask_full), ("hard", mask_hard),
                            ("hcperm", mask_hcperm), ("hc", mask_hc)]:
            sel_uids = npz_uids[mask]
            sel_resp = max_resp[mask]
            rows = []
            weights = []
            missing = 0
            for u, wgt in zip(sel_uids.tolist(), sel_resp.tolist()):
                r = uid_to_row.get(int(u))
                if r is not None:
                    rows.append(r)
                    weights.append(wgt)
                else:
                    missing += 1
            if missing:
                print(f"  WARNING: {missing} UIDs not found in passthrough .cs")
            rows = np.array(rows, dtype=np.intp)
            subset = cs[rows].copy()
            if args.inject_alpha_weight and has_alpha:
                subset["alignments3D/alpha"] = np.asarray(weights, dtype=np.float32)

            out_path = os.path.join(args.outdir,
                                    f"cryodrgn_class_{name}{tag_of[label]}.cs")
            _save_cs(out_path, subset)
            rows_summary.append((name, label, len(rows)))
            counts[label] = len(rows)

        print(f"{name:<8} {counts['full']:>9,} {counts['hard']:>9,} "
              f"{counts['hcperm']:>9,} {counts['hc']:>9,}")

    print("\n[export] wrote .cs files to", args.outdir)
    print("\n--- CryoSPARC import + refinement instructions ---")
    print("For each .cs file:")
    print("  1. CryoSPARC > Import Particles > select the .cs file")
    print("  2. Run Ab-initio Reconstruction (K=1, default settings)")
    print("  3. Use the ab-initio map as input to NU-Refinement")
    print("  full   -> all cryoDRGN-assigned particles (run NU on cryoDRGN particles)")
    print("  hard   -> hard assignments only (cryoDRGN argmax == CryoSPARC argmax)")
    print("  hcperm -> permissive low-uncertainty (keeps ~2x more than strict hc)")
    print("  hc     -> strict low-uncertainty (fewest particles, purest signal)")
    print("\nFiles:")
    for name, label, n in rows_summary:
        fname = f"cryodrgn_class_{name}{tag_of[label]}.cs"
        print(f"  {fname:<40}  {n:>8,} particles")


if __name__ == "__main__":
    main()
