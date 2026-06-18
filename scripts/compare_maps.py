"""Compare cryo-EM density maps: real-space correlation, FSC, and difference maps.

Given two or more ``.mrc`` volumes of the same molecule (e.g. the three J1069
NU-refined conformational classes P6/P7/P8), this quantifies how similar they
are and where they differ:

* Pairwise real-space cross-correlation (CC) inside a shared mask — one number
  per pair (1.0 = identical, lower = more different).
* Pairwise Fourier Shell Correlation (FSC) curves — similarity as a function of
  resolution. Two maps of the same molecule are ~1.0 at low resolution; an
  early drop-off means they diverge in a *structured* way (real conformational
  difference), not just high-frequency noise.
* Difference maps (map_j - map_i), written as ``.mrc`` for visualisation in
  ChimeraX, plus the RMS of each difference as a scalar summary.

All maps must share the same box size and voxel size (checked at load).

Usage
-----
    python scripts/compare_maps.py \
        --maps data/maps/cryosparc_P25_J1076_010_volume_map.mrc \
               data/maps/cryosparc_P25_J1077_010_volume_map.mrc \
               data/maps/cryosparc_P25_J1078_011_volume_map.mrc \
        --labels P6 P7 P8 \
        --outdir results_J1069/map_comparison
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import mrcfile
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--maps", required=True, nargs="+", help="Input .mrc volumes")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Short label per map (default: file stem)")
    p.add_argument("--mask-percentile", type=float, default=85.0,
                   help="Voxels above this percentile of the mean map define "
                        "the shared molecule mask for real-space CC (default 85)")
    p.add_argument("--fsc-threshold", type=float, default=0.5,
                   help="FSC threshold to report a divergence resolution (0.5)")
    p.add_argument("--outdir", default="results/map_comparison")
    return p.parse_args()


def load_map(path: str):
    with mrcfile.open(path, permissive=True) as m:
        data = np.asarray(m.data, dtype=np.float32).copy()
        vox = float(m.voxel_size.x)
    return data, vox


def radial_fsc(a: np.ndarray, b: np.ndarray):
    """Fourier Shell Correlation between two equal-sized real-space maps.

    Returns (freq_cycles_per_voxel, fsc). Multiply freq by 1/voxel to get
    cycles/Angstrom; resolution = 1 / (freq / voxel) = voxel / freq.
    """
    Fa = np.fft.fftshift(np.fft.fftn(a))
    Fb = np.fft.fftshift(np.fft.fftn(b))

    n = a.shape[0]
    c = n // 2
    z, y, x = np.indices(a.shape)
    r = np.sqrt((z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2)
    r = r.astype(np.int32)

    nbins = c
    num = np.zeros(nbins)
    da = np.zeros(nbins)
    db = np.zeros(nbins)
    cross = Fa * np.conj(Fb)
    for i in range(nbins):
        shell = r == i
        if not shell.any():
            continue
        num[i] = np.real(cross[shell].sum())
        da[i] = np.abs(Fa[shell]).__pow__(2).sum()
        db[i] = np.abs(Fb[shell]).__pow__(2).sum()
    denom = np.sqrt(da * db)
    fsc = np.divide(num, denom, out=np.zeros_like(num), where=denom > 0)
    freq = np.arange(nbins) / n          # cycles per voxel
    return freq, fsc


def masked_cc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    av = a[mask].ravel()
    bv = b[mask].ravel()
    av = av - av.mean()
    bv = bv - bv.mean()
    denom = np.sqrt((av ** 2).sum() * (bv ** 2).sum())
    return float((av * bv).sum() / denom) if denom > 0 else 0.0


def main():
    args = parse_args()
    out = Path(args.outdir)
    (out / "difference_maps").mkdir(parents=True, exist_ok=True)

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
        print(f"      {lab}: shape={d.shape}  voxel={v:.3f} A  "
              f"min/max={d.min():.3g}/{d.max():.3g}")
    shapes = {m.shape for m in maps}
    if len(shapes) != 1:
        raise SystemExit(f"Maps have differing box sizes: {shapes}")
    vox = voxels[0]
    if max(voxels) - min(voxels) > 1e-3:
        print(f"      WARNING: voxel sizes differ {voxels}; using {vox:.3f}")

    print(f"[2/4] Building shared mask (>{args.mask_percentile:.0f}th pct of mean map)")
    mean_map = np.mean(maps, axis=0)
    thresh = np.percentile(mean_map, args.mask_percentile)
    mask = mean_map > thresh
    print(f"      mask covers {100 * mask.mean():.1f}% of the box "
          f"({int(mask.sum()):,} voxels)")

    pairs = list(itertools.combinations(range(len(maps)), 2))

    print("[3/4] Real-space cross-correlation + difference maps")
    cc_matrix = np.eye(len(maps))
    rows = []
    for i, j in pairs:
        cc = masked_cc(maps[i], maps[j], mask)
        cc_matrix[i, j] = cc_matrix[j, i] = cc
        diff = maps[j] - maps[i]
        diff_rms = float(np.sqrt((diff[mask] ** 2).mean()))
        dname = out / "difference_maps" / f"diff_{labels[j]}_minus_{labels[i]}.mrc"
        with mrcfile.new(str(dname), overwrite=True) as m:
            m.set_data(diff.astype(np.float32))
            m.voxel_size = vox
        rows.append({"pair": f"{labels[i]} vs {labels[j]}",
                     "cross_correlation": round(cc, 4),
                     "diff_rms": round(diff_rms, 5),
                     "diff_map": dname.name})
        print(f"      {labels[i]} vs {labels[j]}:  CC={cc:.4f}  "
              f"diff_RMS={diff_rms:.4g}  -> {dname.name}")

    import csv
    with open(out / "map_correlation_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pair", "cross_correlation",
                                           "diff_rms", "diff_map"])
        w.writeheader()
        w.writerows(rows)

    print("[4/4] Fourier Shell Correlation curves")
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    fsc_table = {}
    for i, j in pairs:
        freq, fsc = radial_fsc(maps[i], maps[j])
        res = np.full_like(freq, np.inf)
        nz = freq > 0
        res[nz] = vox / freq[nz]               # Angstrom per the Nyquist scaling
        fsc_table[f"{labels[i]}_vs_{labels[j]}"] = fsc
        ax.plot(freq / vox, fsc, lw=1.8, label=f"{labels[i]} vs {labels[j]}")

        # resolution where FSC first drops below threshold
        below = np.where((fsc < args.fsc_threshold) & nz)[0]
        if len(below):
            r_cross = vox / freq[below[0]]
            print(f"      {labels[i]} vs {labels[j]}: FSC<{args.fsc_threshold} "
                  f"first at ~{r_cross:.1f} A")

    ax.axhline(args.fsc_threshold, color="0.6", ls="--", lw=1,
               label=f"FSC={args.fsc_threshold}")
    ax.axhline(0.143, color="0.8", ls=":", lw=1, label="FSC=0.143")
    ax.set_xlabel("Spatial frequency (1/Å)")
    ax.set_ylabel("Fourier Shell Correlation")
    ax.set_ylim(-0.1, 1.05)
    # start just above 0 so the resolution (1/f) secondary axis stays finite
    nyquist = 1.0 / (2.0 * vox)
    ax.set_xlim(1.0 / (maps[0].shape[0] * vox), nyquist)
    ax.set_title("Pairwise FSC between conformational maps\n"
                 "(high = similar; early drop = real structural difference)",
                 fontsize=10, pad=32)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    # secondary x-axis in resolution (Angstrom)
    def f2r(f):
        f = np.asarray(f, dtype=float)
        out_ = np.full_like(f, np.nan)
        nz = f > 1e-6
        out_[nz] = 1.0 / f[nz]
        return out_
    secax = ax.secondary_xaxis("top", functions=(f2r, f2r))
    secax.set_xlabel("Resolution (Å)")
    # explicit, non-overlapping resolution ticks
    res_ticks = [20, 10, 6, 4, 3]
    secax.set_xticks([1.0 / r for r in res_ticks])
    secax.set_xticklabels([str(r) for r in res_ticks])

    fig.tight_layout()
    fig.savefig(out / "fsc_curves.png", dpi=150, bbox_inches="tight")

    # FSC table to CSV (freq + resolution + each pair)
    freq, _ = radial_fsc(maps[0], maps[1])
    with open(out / "fsc_curves.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        header = ["freq_per_A", "resolution_A"] + list(fsc_table.keys())
        w.writerow(header)
        for idx in range(len(freq)):
            f_per_a = freq[idx] / vox
            res_a = (1.0 / f_per_a) if f_per_a > 0 else ""
            w.writerow([f"{f_per_a:.5f}", res_a if res_a == "" else f"{res_a:.2f}"]
                       + [f"{fsc_table[k][idx]:.4f}" for k in fsc_table])

    print(f"\nDone. Outputs in {out.resolve()}:")
    print("  map_correlation_summary.csv  (CC + diff RMS per pair)")
    print("  fsc_curves.png / .csv        (resolution-dependent similarity)")
    print("  difference_maps/*.mrc        (open in ChimeraX at +/- 2-3x RMS)")


if __name__ == "__main__":
    main()
