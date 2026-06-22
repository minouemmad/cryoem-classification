"""Map-level validation for one classification branch.

Two literature-standard map tests, computed directly from the CryoSPARC half
maps and full maps in the branch output tree:

1. GOLD-STANDARD RESOLUTION (Scheres & Chen 2012; Rosenthal & Henderson 2003)
   FSC(half_A, half_B) per class -> resolution at FSC=0.143. This is the honest
   per-class resolution the refinement achieved.

2. CROSS-CLASS SIMILARITY (the "are these really different states?" test)
   FSC(class_i, class_j) and masked real-space CC between the per-class maps.
   * For Homogeneous RECONSTRUCTION the three classes were back-projected with a
     shared (J1442) pose frame, so the maps are directly comparable -- this is
     the clean cross-class comparison.
   * For independent refinements (NU / Homog refinement) each class re-optimised
     its own pose frame, so a raw cross-class FSC is contaminated by global
     misalignment; those rows are computed but flagged ``needs_alignment=True``.

   Decision rule: if the cross-class FSC falls BELOW each class's own gold-
   standard FSC at resolutions where both maps are individually resolved, the
   difference is real structure; if they agree to their own resolution limit the
   classes are the same state split arbitrarily.

Run
---
    python scripts/analyze_branch_maps.py \
        --branch-dir results_J1069/cryosparc_outputs/with_1442_weights \
        --outdir results_J1069/branch_validation_w1442/maps
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.compare_maps import load_map, radial_fsc, masked_cc

LABELS = ["P6", "P7", "P8"]
_COLORS = {"P6": "#2ca02c", "P7": "#d62728", "P8": "#1f77b4"}

# job_key -> (subdir, {label: (full_map, half_A, half_B)}) ; filled by discovery
JOBS = {
    "homo_reconstruction": ("homogeneous_reconstruction",
                            {"P6": "J3556", "P7": "J3557", "P8": "J3558"}, False),
    "nu_refinement":       ("non-uniform_refinement",
                            {"P6": "J3562", "P7": "J3563", "P8": "J3564"}, True),
    "homo_refinement":     ("homogeneous_refinement",
                            {"P6": "J3565", "P7": "J3566", "P8": "J3567"}, True),
}


def _find(subdir: Path, job_id: str, suffix: str):
    # job_id may be followed by an optional class-index segment (e.g. _006)
    hits = [p for p in subdir.glob(f"cryosparc_P25_{job_id}*{suffix}")
            if p.name.endswith(suffix)]
    return hits[0] if hits else None


def fsc_resolution(freq, fsc, vox, thr):
    """First resolution (A) where fsc drops below thr. freq in cyc/voxel."""
    below = np.where(fsc < thr)[0]
    if len(below) == 0:
        return None
    i = below[0]
    if i == 0:
        return None
    # linear interp in freq for the crossing
    f0, f1 = freq[i - 1], freq[i]
    s0, s1 = fsc[i - 1], fsc[i]
    fc = f0 + (thr - s0) * (f1 - f0) / (s1 - s0) if s1 != s0 else f1
    return float(vox / fc) if fc > 0 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--branch-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--mask-percentile", type=float, default=85.0)
    args = ap.parse_args()

    branch = Path(args.branch_dir)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    gold_rows = []
    cross_rows = []

    for job, (subdir_name, ids, needs_align) in JOBS.items():
        print(f"\n=== {job} ===")
        full_maps, vox_job = {}, None
        # ---- gold-standard FSC per class ----
        for lab in LABELS:
            cdir = branch / subdir_name / lab
            if not cdir.exists():
                print(f"  {lab}: dir missing, skip")
                continue
            jid = ids[lab]
            full = _find(cdir, jid, "_volume_map.mrc")
            hA = _find(cdir, jid, "_volume_map_half_A.mrc")
            hB = _find(cdir, jid, "_volume_map_half_B.mrc")
            if full is None or hA is None or hB is None:
                print(f"  {lab}: maps missing (full={full} hA={hA} hB={hB})")
                continue
            mA, vox = load_map(str(hA))
            mB, _ = load_map(str(hB))
            mF, _ = load_map(str(full))
            vox_job = vox
            full_maps[lab] = mF
            freq, fsc = radial_fsc(mA, mB)
            res143 = fsc_resolution(freq, fsc, vox, 0.143)
            res50 = fsc_resolution(freq, fsc, vox, 0.5)
            gold_rows.append({"job": job, "class": lab,
                              "gold_res_0.143_A": res143,
                              "gold_res_0.5_A": res50,
                              "box": mA.shape[0], "voxel_A": vox})
            # save curve
            pd.DataFrame({"freq_cyc_per_vox": freq, "fsc": fsc,
                          "res_A": np.divide(vox, freq, out=np.full_like(freq, np.inf),
                                             where=freq > 0)}).to_csv(
                out / f"goldfsc_{job}_{lab}.csv", index=False)
            print(f"  {lab}: gold-standard {res143:.2f} A (0.143)" if res143
                  else f"  {lab}: gold-standard res N/A")

        # ---- cross-class FSC + CC ----
        if len(full_maps) >= 2:
            labs = list(full_maps.keys())
            # shared mask from mean of the per-class full maps
            stack = np.stack([full_maps[l] for l in labs])
            meanmap = stack.mean(axis=0)
            thr = np.percentile(meanmap, args.mask_percentile)
            mask = meanmap > thr
            fig, ax = plt.subplots(figsize=(6, 4.2))
            for li, lj in itertools.combinations(labs, 2):
                freq, fsc = radial_fsc(full_maps[li], full_maps[lj])
                cc = masked_cc(full_maps[li], full_maps[lj], mask)
                res50 = fsc_resolution(freq, fsc, vox_job, 0.5)
                cross_rows.append({"job": job, "pair": f"{li}-{lj}",
                                   "cross_CC": cc,
                                   "cross_FSC0.5_res_A": res50,
                                   "needs_alignment": needs_align})
                res_axis = np.divide(vox_job, freq, out=np.full_like(freq, np.inf),
                                     where=freq > 0)
                ax.plot(freq, fsc, label=f"{li}-{lj} (CC={cc:.2f})")
                print(f"  cross {li}-{lj}: CC={cc:.3f}  FSC0.5 res="
                      f"{res50:.1f} A" if res50 else
                      f"  cross {li}-{lj}: CC={cc:.3f}  FSC0.5 res=N/A"
                      + ("  [needs alignment]" if needs_align else ""))
            ax.axhline(0.5, color="grey", ls="--", lw=0.8)
            ax.axhline(0.143, color="grey", ls=":", lw=0.8)
            ax.set_xlabel("spatial frequency (cycles/voxel)")
            ax.set_ylabel("FSC")
            ax.set_ylim(-0.1, 1.05)
            ax.set_title(f"{job} cross-class FSC"
                         + ("  (UNALIGNED — interpret low-res only)" if needs_align else ""))
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out / f"crossfsc_{job}.png", dpi=160)
            plt.close(fig)

    pd.DataFrame(gold_rows).to_csv(out / "gold_standard_resolution.csv", index=False)
    pd.DataFrame(cross_rows).to_csv(out / "cross_class_similarity.csv", index=False)
    print(f"\nDone. Map validation outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
