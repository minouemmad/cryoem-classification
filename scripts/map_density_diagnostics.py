"""Density / occupancy diagnostics for a set of ALIGNED cryo-EM maps.

This complements compare_maps.py. CC and FSC answer "how similar in *shape*"
but they are (deliberately) blind to two things you also need to judge a
classification:

  1. OCCUPANCY / local density amount -- a conformational class can differ from
     another not by a rigid-body motion but by a region being ORDERED vs
     disordered/absent (e.g. a flexible domain, a bound ligand). That shows up
     as a difference in density *magnitude* in a localised region, which CC
     (mean-subtracted, scale-invariant) and FSC largely ignore.

  2. Whether an observed difference is LOCALISED (a real, interpretable change)
     or DELOCALISED (smeared everywhere -> consistent with noise / resolution
     mismatch -> classes are really the same = over-splitting).

IMPORTANT scaling caveat
------------------------
Independently refined maps each carry an ARBITRARY global amplitude scale (and
different B-factor sharpening), so you may NOT compare absolute density between
them directly. Here we remove that scale by normalising every map so that the
mean density inside a CONSENSUS CORE (voxels strong in *all* maps -- assumed to
be the rigid, always-present core) equals 1.0. Occupancy comparisons are then
expressed relative to that shared core.

Metrics written (per map, and per pair) to a CSV:
  * core_norm_factor         -- the per-map scale that was divided out.
  * integrated_density_mask  -- total (core-normalised) density inside the
                                shared molecule mask = an occupancy/"mass" proxy.
  * volume_vox_at_core_half  -- # voxels above 0.5x the core level = molecular
                                volume proxy at a common, scale-free contour.
  * pair CC (Pearson, masked) -- shape similarity (same as compare_maps.py).
  * pair diff_localization    -- fraction of the total squared difference held
                                by the top 5% |difference| voxels. High (-> ~1)
                                = localised, structured change. Low (-> 0.05,
                                i.e. uniform) = diffuse, noise-like.
  * pair diff_rms             -- magnitude of the (core-normalised) difference.

Usage
-----
    python scripts/map_density_diagnostics.py \
        --maps data/maps/J1069_homog_aligned/J1076_P6.mrc ... \
        --labels P6 P7 P8 \
        --outdir results_J1069/density_diagnostics
"""
from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import mrcfile
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--maps", required=True, nargs="+")
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--mask-percentile", type=float, default=85.0,
                   help="shared molecule mask = voxels above this pct of mean map")
    p.add_argument("--core-percentile", type=float, default=95.0,
                   help="consensus core = voxels above this pct in EVERY map "
                        "(used for scale normalisation)")
    p.add_argument("--top-frac", type=float, default=0.05,
                   help="fraction of voxels used for the difference-localisation metric")
    p.add_argument("--outdir", default="results/density_diagnostics")
    return p.parse_args()


def load_map(path: str):
    with mrcfile.open(path, permissive=True) as m:
        data = np.asarray(m.data, dtype=np.float32).copy()
        vox = float(m.voxel_size.x)
    return data, vox


def diff_localization(diff: np.ndarray, mask: np.ndarray, top_frac: float) -> float:
    """Fraction of total squared difference held by the top `top_frac` voxels.

    Uniform (pure noise) -> ~top_frac. Fully localised -> ~1.0.
    """
    d2 = (diff[mask].ravel()) ** 2
    total = d2.sum()
    if total <= 0:
        return 0.0
    k = max(1, int(round(top_frac * d2.size)))
    top = np.partition(d2, d2.size - k)[d2.size - k:]
    return float(top.sum() / total)


def main():
    args = parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    paths = args.maps
    labels = args.labels or [Path(p).stem for p in paths]
    if len(labels) != len(paths):
        raise SystemExit("--labels count must match --maps count")

    print(f"[1/4] Loading {len(paths)} maps")
    maps, voxels = [], []
    for p, lab in zip(paths, labels):
        d, v = load_map(p)
        maps.append(d)
        voxels.append(v)
        print(f"      {lab}: shape={d.shape} voxel={v:.3f} A")
    shapes = {m.shape for m in maps}
    if len(shapes) != 1:
        raise SystemExit(f"Maps differ in box size: {shapes}; align/resample first.")
    vox = voxels[0]

    print(f"[2/4] Shared mask (>{args.mask_percentile} pct of mean) "
          f"+ consensus core (>{args.core_percentile} pct in EVERY map)")
    mean_map = np.mean(maps, axis=0)
    mask = mean_map > np.percentile(mean_map, args.mask_percentile)
    core = np.ones(maps[0].shape, dtype=bool)
    for d in maps:
        core &= d > np.percentile(d, args.core_percentile)
    print(f"      mask covers {mask.mean()*100:.1f}% of box, "
          f"consensus core {core.mean()*100:.2f}% ({int(core.sum()):,} voxels)")
    if core.sum() < 50:
        raise SystemExit("Consensus core too small; lower --core-percentile.")

    print("[3/4] Core-normalising each map (removes arbitrary per-refinement scale)")
    norm_maps, per_map_rows = [], []
    for lab, d in zip(labels, maps):
        scale = float(d[core].mean())
        nd = d / scale if scale != 0 else d
        norm_maps.append(nd)
        integ = float(nd[mask].sum())
        vol_vox = int((nd > 0.5).sum())  # voxels above half the core level
        per_map_rows.append({
            "label": lab,
            "core_norm_factor": scale,
            "integrated_density_mask": integ,
            "volume_vox_at_core_half": vol_vox,
            "volume_A3_at_core_half": vol_vox * vox**3,
        })
        print(f"      {lab}: scale={scale:.4g}  integ_density(mask)={integ:.4g}  "
              f"vol@0.5core={vol_vox*vox**3:.3e} A^3")

    print("[4/4] Pairwise difference localisation + magnitude (core-normalised)")
    pair_rows = []
    for (i, li), (j, lj) in itertools.combinations(enumerate(labels), 2):
        a, b = norm_maps[i], norm_maps[j]
        av = a[mask].ravel() - a[mask].mean()
        bv = b[mask].ravel() - b[mask].mean()
        denom = np.sqrt((av**2).sum() * (bv**2).sum())
        cc = float((av*bv).sum()/denom) if denom > 0 else 0.0
        diff = b - a
        loc = diff_localization(diff, mask, args.top_frac)
        rms = float(np.sqrt((diff[mask]**2).mean()))
        pair_rows.append({
            "pair": f"{li} vs {lj}",
            "cc_pearson": round(cc, 4),
            "diff_localization_top%d%%" % int(args.top_frac*100): round(loc, 4),
            "diff_rms_corenorm": round(rms, 5),
        })
        print(f"      {li} vs {lj}: CC={cc:.3f}  "
              f"localization(top{int(args.top_frac*100)}%)={loc:.3f}  diffRMS={rms:.4g}")

    with open(out / "per_map_density.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_map_rows[0].keys()))
        w.writeheader(); w.writerows(per_map_rows)
    with open(out / "pairwise_density.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        w.writeheader(); w.writerows(pair_rows)

    print(f"\nDone. Wrote per_map_density.csv and pairwise_density.csv to {out.resolve()}")
    print("Interpretation: uniform/noise localization ~= top-fraction (e.g. 0.05); "
          "localized real change -> toward 1.0.")


if __name__ == "__main__":
    main()
