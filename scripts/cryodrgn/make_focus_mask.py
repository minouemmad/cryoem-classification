#!/usr/bin/env python
"""Build a 3D FOCUS MASK (mobile region) + consensus volume for focused cryoDRGN.

Focused heterogeneity needs two things defined in 3D (NOT 2D -- the moving part
lands at a different image location in every particle depending on pose):

  * consensus.mrc  : the mean volume across a set of cryoDRGN-decoded states
                     (= the "rigid + everything" reference to be subtracted).
  * focus_mask.mrc : a soft 0..1 mask over the region that MOVES the most,
                     found as the high-variance voxels across those states.

Feed a set of volumes that span the motion -- e.g. the kmeans20 volumes from
``cryodrgn analyze`` (analyze.N/kmeans20/vol_*.mrc) or PC1/PC2 traversal volumes.
The per-voxel standard deviation across them is the data-driven "what moves" map;
this is cryoDRGN's own answer to your step 1 ("which regions move the most").

Run with the cryodrgn env (or cryodrgn-py310) from repo root::

    python scripts/cryodrgn/make_focus_mask.py \\
      --volumes "results_cryodrgn/J1442_gP25_WT_POSE_BIAS/train_recover_D128_z16_b0p03/analyze.50/kmeans20/vol_*.mrc" \\
      --apix 2.075 --mask-quantile 0.90 \\
      -o results_cryodrgn/J1442_gP25_WT_POSE_BIAS/focus
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from scipy.ndimage import binary_dilation
from cryodrgn.mrcfile import parse_mrc, write_mrc
from cryodrgn.masking import cosine_dilation_mask


def build_mask(consensus, varmap, apix, protein_quantile, mask_quantile,
               dilate_a, edge_a):
    """Tight, protein-CLAMPED mask of the highest-variance sub-region.

    Returns (mask, info-dict).  The mask is forced to stay on/near the molecule
    so it can never balloon into the solvent halo.
    """
    prot_level = float(np.quantile(consensus, protein_quantile))
    protein = consensus > prot_level                     # tight molecular envelope
    var_thresh = float(np.quantile(varmap[protein], mask_quantile))
    seed = ((varmap >= var_thresh) & protein).astype(np.float32)

    mask = cosine_dilation_mask(seed, threshold=0.5, dilation=dilate_a,
                                edge_dist=edge_a, apix=apix, verbose=False)
    mask = np.asarray(mask, dtype=np.float32)

    # CLAMP to the protein dilated by just the soft-edge width so the mask
    # follows the moving domain but does not leak across the solvent
    grow = int(np.ceil((dilate_a + edge_a) / apix)) + 1
    bound = binary_dilation(protein, iterations=grow)
    mask *= bound.astype(np.float32)

    info = dict(prot_level=prot_level, protein_frac=float(protein.mean()),
                var_thresh=var_thresh, mask_box_frac=float((mask > 0.5).mean()),
                mask_of_protein=float(((mask > 0.5) & protein).sum()
                                      / max(protein.sum(), 1)))
    return mask, protein, info


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--volumes", help="glob for the decoded state volumes (quote it)")
    ap.add_argument("--consensus", help="reuse an existing consensus.mrc (skip --volumes)")
    ap.add_argument("--variance", help="reuse an existing variance.mrc (skip --volumes)")
    ap.add_argument("--apix", type=float, required=True,
                    help="pixel size of the volumes (e.g. 2.075 for J1442 D=128)")
    ap.add_argument("--mask-quantile", type=float, default=0.90,
                    help="within the protein, voxels above this variance quantile "
                         "seed the mobile region (default 0.90; raise to tighten)")
    ap.add_argument("--protein-quantile", type=float, default=0.985,
                    help="consensus quantile defining the MOLECULE (default 0.985 "
                         "= tight envelope; lower = looser). NOTE a mean-zero "
                         "decoded map needs a HIGH quantile or the envelope "
                         "swallows the solvent")
    ap.add_argument("--dilate", type=int, default=6,
                    help="dilation of the mobile region in ANGSTROMS (default 6)")
    ap.add_argument("--edge", type=int, default=4,
                    help="soft cosine edge width in ANGSTROMS (default 4)")
    ap.add_argument("-o", "--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.consensus and args.variance:
        consensus = np.asarray(parse_mrc(args.consensus)[0], dtype=np.float32)
        varmap = np.asarray(parse_mrc(args.variance)[0], dtype=np.float32)
        print(f"[load] reusing consensus + variance (box {consensus.shape[-1]})")
    else:
        if not args.volumes:
            raise SystemExit("give --volumes, or --consensus + --variance")
        paths = sorted(glob.glob(args.volumes))
        if len(paths) < 2:
            raise SystemExit(f"need >=2 volumes, matched {len(paths)} for {args.volumes}")
        print(f"[load] {len(paths)} volumes")
        vols = [np.asarray(parse_mrc(p)[0], dtype=np.float32) for p in paths]
        if len({v.shape for v in vols}) != 1:
            raise SystemExit("volumes have inconsistent shapes")
        V = np.stack(vols, 0)
        consensus, varmap = V.mean(0), V.std(0)

    D = consensus.shape[-1]
    mask, protein, info = build_mask(
        consensus, varmap, args.apix, args.protein_quantile,
        args.mask_quantile, args.dilate, args.edge)

    print(f"[stat] box {D}  protein envelope = {info['protein_frac']*100:.2f}% of box "
          f"(consensus > {info['prot_level']:.4f})")
    print(f"[mask] mobile mask = {info['mask_box_frac']*100:.2f}% of box, "
          f"covering {info['mask_of_protein']*100:.0f}% of the protein "
          f"(variance > {info['var_thresh']:.4f})")
    if info["mask_of_protein"] > 0.85:
        print("  WARNING: mask covers almost the WHOLE protein -> not focused. "
              "Raise --mask-quantile (e.g. 0.96) and/or --protein-quantile, or "
              "supply a hand-drawn ChimeraX mask instead.")

    cpath = os.path.join(args.outdir, "consensus.mrc")
    mpath = os.path.join(args.outdir, "focus_mask.mrc")
    write_mrc(cpath, consensus.astype(np.float32), Apix=args.apix, is_vol=True)
    write_mrc(mpath, mask.astype(np.float32), Apix=args.apix, is_vol=True)
    write_mrc(os.path.join(args.outdir, "variance.mrc"),
              varmap.astype(np.float32), Apix=args.apix, is_vol=True)
    print(f"[out] {cpath}\n      {mpath}\n      {args.outdir}/variance.mrc")
    print("In ChimeraX: open both, then `volume #<mask> level 0.5` (NOT "
          "fastEncloseVolume) to see the true footprint; consensus ~level "
          f"{info['prot_level']:.3f}. Aim for a mask on ONE moving domain "
          "(~20-60% of the protein).")


if __name__ == "__main__":
    main()
