#!/usr/bin/env python
"""Compare reconstructed maps head-to-head: cryoDRGN-basin NU maps vs the best
CryoSPARC hetero-refine class maps (and sub-hetero maps of unknown class).

Answers the four questions the user asked, for each dataset group:

  1. RIGID-BODY ALIGNMENT  - every map is reconstructed in its OWN pose frame,
     so we first bring them all into one common frame (coarse global rotation
     search + Powell refine, maximising masked real-space CC).  Reported as the
     rotation/shift needed to align each map to the reference.
  2. FSC                    - Fourier-shell-correlation between every pair of
     aligned maps; the 0.5 and 0.143 crossings are the "resolution of
     agreement" (how far into high resolution two maps still match).
  3. LOCAL RESOLUTION       - true per-map local resolution needs the two
     unfiltered half-maps (not uploaded here).  With only sharpened full maps we
     instead compute a LOCAL CROSS-CORRELATION map (sliding boxcar Pearson
     correlation between a matched cryoDRGN/CryoSPARC pair) = "where do the two
     maps agree vs disagree", which is the map-comparison analogue and pinpoints
     the region that carries the difference.  Clearly labelled as such.
  4. VISUAL DIFFERENCES     - central-slice montages of every map + signed
     difference maps for each matched pair.

It also IDENTIFIES which class each unknown sub-hetero map belongs to, purely
from image similarity (best masked CC / FSC to the labelled set).

All maps must cover the same physical field of view (box*apix); they are
resampled to a common working box before comparison.

Run from repo root with the cryodrgn-py310 python::

    python scripts/cryodrgn/compare_maps.py \
      --map "J1442:cryodrgn:basin1:cryosparc_P25_J1442_basin_1_particles_volume_map_sharp.mrc" \
      ... \
      -o results_cryodrgn/map_comparison
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter, zoom

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for p in (_REPO, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cryodrgn.mrcfile import parse_mrc

from cryodrgn_decode_states import cross_correlation, read_mrc
from cryodrgn_lda_states import fsc_curve, normalise_in_mask
from cryodrgn_focused_analysis import align_rigid


# --------------------------------------------------------------------------- #
def map_apix(path):
    _, h = parse_mrc(path)
    return float(h.apix)


def resample_box(vol, n):
    if vol.shape[0] == n:
        return vol.astype(np.float32)
    return zoom(vol, n / vol.shape[0], order=1).astype(np.float32)


def solid_mask(ref):
    m = ref > (ref.mean() + 0.5 * ref.std())
    if m.sum() < 100:
        m = ref > ref.mean()
    return m


def fsc_res(v0, v1, apix):
    freq, fsc, res05 = fsc_curve(v0, v1, apix)
    below = np.where(fsc < 0.143)[0]
    res143 = float(1.0 / freq[below[0]]) if below.size and below[0] > 0 else np.inf
    return freq, fsc, res05, res143


def lowpass(v, apix, res_A):
    """Gaussian low-pass to ~res_A; makes CC/alignment robust to sharpening noise."""
    if not res_A:
        return v
    sigma = max(res_A / (2.0 * apix), 0.5)
    return gaussian_filter(v.astype(np.float32), sigma)


def masked_ssim(a, b, mask):
    """Global structural-similarity index (SSIM) over masked voxels. A
    non-resolution metric: rewards matching local means/contrast/covariance,
    1.0 = identical, ~0 = unrelated."""
    x = a[mask].astype(np.float64)
    y = b[mask].astype(np.float64)
    L = max(x.max() - x.min(), y.max() - y.min(), 1e-8)
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mx) * (y - my)).mean()
    return float(((2 * mx * my + c1) * (2 * cov + c2)) /
                 ((mx * mx + my * my + c1) * (vx + vy + c2)))


def masked_nmi(a, b, mask, bins=64):
    """Normalised mutual information of the two maps' intensities inside the mask
    (non-resolution, non-linear dependence). 0 = independent, 1 = identical."""
    x, y = a[mask], b[mask]
    hist, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = hist / hist.sum()
    px, py = pxy.sum(1), pxy.sum(0)
    nz = pxy > 0
    hxy = -(pxy[nz] * np.log(pxy[nz])).sum()
    hx = -(px[px > 0] * np.log(px[px > 0])).sum()
    hy = -(py[py > 0] * np.log(py[py > 0])).sum()
    mi = hx + hy - hxy
    return float(2 * mi / (hx + hy)) if (hx + hy) > 0 else 0.0


def compare_to_ref(ref, mov, apix, align_box, n_rot, lp_A):
    """Rigid-body align `mov` onto `ref` (pairwise) then score their agreement.
    Returns aligned mov + metrics.  Score = mean FSC over shells out to 1/8 A
    (robust similarity for assignment); CC is measured on low-passed maps so it
    is not dominated by high-frequency sharpening noise."""
    aligned, info = align_rigid(ref, mov, n_search=align_box, n_rot=n_rot, seed=0)
    mask = solid_mask(ref)
    cc = cross_correlation(lowpass(aligned, apix, lp_A),
                           lowpass(ref, apix, lp_A), mask)
    freq, fsc, res05, res143 = fsc_res(aligned, ref, apix)
    band = (freq > 0) & (freq <= 1.0 / 8.0)
    score = float(np.mean(fsc[band])) if band.any() else float(np.mean(fsc[freq > 0]))
    return {"aligned": aligned, "info": info, "cc": cc, "res05": res05,
            "res143": res143, "score": score, "freq": freq, "fsc": fsc,
            "mask": mask}


def local_corr_map(a, b, win):
    """Sliding-boxcar local Pearson correlation between two aligned volumes.

    Fast, vectorised (uniform_filter).  Value near 1 = the two maps agree in
    that neighbourhood; low/negative = they diverge there.  This is a
    map-vs-map local agreement proxy, NOT ResMap-style local resolution (which
    needs half-maps)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mu_a = uniform_filter(a, win)
    mu_b = uniform_filter(b, win)
    va = uniform_filter(a * a, win) - mu_a ** 2
    vb = uniform_filter(b * b, win) - mu_b ** 2
    cov = uniform_filter(a * b, win) - mu_a * mu_b
    denom = np.sqrt(np.clip(va, 0, None) * np.clip(vb, 0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        lcc = np.where(denom > 1e-8, cov / denom, np.nan)
    return lcc.astype(np.float32)


def write_mrc(path, vol, apix):
    from cryodrgn.mrcfile import write_mrc as _w
    try:
        _w(path, vol.astype(np.float32), Apix=float(apix), is_vol=True)
    except Exception:
        _w(path, vol.astype(np.float32))


# --------------------------------------------------------------------------- #
def load_group(specs, work_box):
    """specs: list of (method, name, path).  Returns list of map dicts resampled
    to `work_box`, plus the common apix."""
    maps = []
    phys = None
    for method, name, path in specs:
        if not os.path.isfile(path):
            print(f"  [skip] missing {path}")
            continue
        vol = read_mrc(path)
        ap = map_apix(path)
        ph = vol.shape[0] * ap
        phys = ph if phys is None else phys
        if abs(ph - phys) > 1.0:
            print(f"  [warn] {name}: physical size {ph:.1f} != {phys:.1f} A")
        maps.append({"method": method, "name": name, "path": path,
                     "orig_box": int(vol.shape[0]), "orig_apix": ap,
                     "vol": resample_box(vol, work_box)})
    work_apix = phys / work_box if phys else None
    return maps, work_apix, phys


def align_group(maps, ref_idx, n_rot, align_box):
    ref = maps[ref_idx]["vol"]
    maps[ref_idx]["aligned"] = ref
    maps[ref_idx]["align"] = {"rot_deg": 0.0, "shift_vox": 0.0,
                              "cc_before": 1.0, "cc_after": 1.0}
    for i, m in enumerate(maps):
        if i == ref_idx:
            continue
        a, info = align_rigid(ref, m["vol"], n_search=align_box,
                              n_rot=n_rot, seed=0)
        m["aligned"] = a
        m["align"] = {"rot_deg": info["rot_deg"],
                      "shift_vox": float(np.linalg.norm(info["shift_vox"])),
                      "cc_before": info["cc_raw_lowres"],
                      "cc_after": info["cc_aligned_lowres"]}
        print(f"    align {m['name']:10s} -> {maps[ref_idx]['name']}: "
              f"rot {info['rot_deg']:5.1f} deg  CC "
              f"{info['cc_raw_lowres']:.2f}->{info['cc_aligned_lowres']:.2f}")
    return maps


# --------------------------------------------------------------------------- #
def slice_montage(maps, out, label, anchor):
    """Central slices of every map, each pairwise-aligned to `anchor` for display."""
    n = len(maps)
    ncol = min(n, 6)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.7 * nrow),
                             squeeze=False)
    for k, m in enumerate(maps):
        ax = axes[k // ncol][k % ncol]
        v = m.get("disp", m["vol"])
        mid = v.shape[0] // 2
        ax.imshow(v[mid], cmap="gray")
        ax.set_title(f"{m['method']}\n{m['name']}", fontsize=8)
        ax.axis("off")
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(f"{label}: central slices (each aligned to {anchor} for display)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = os.path.join(out, f"{label}_slices.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"  [plot] {p}")


def matrix_heatmap(mat, names_row, names_col, out, label, title, fmt="{:.2f}",
                   vmin=None, vmax=None, cmap="viridis", invert=False):
    nr, nc = len(names_row), len(names_col)
    fig, ax = plt.subplots(figsize=(0.7 * nc + 3, 0.7 * nr + 2.5))
    data = mat.copy()
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(nc)); ax.set_yticks(range(nr))
    ax.set_xticklabels(names_col, rotation=90, fontsize=8)
    ax.set_yticklabels(names_row, fontsize=8)
    for i in range(nr):
        for j in range(nc):
            val = data[i, j]
            if np.isfinite(val):
                ax.text(j, i, fmt.format(val), ha="center", va="center",
                        color="w" if (im.norm(val) < 0.5) ^ invert else "k",
                        fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    p = os.path.join(out, f"{label}.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"  [plot] {p}")


def pair_deepdive(mover, cs, aligned_mov, ref_vol, apix, win, out, tag,
                  res05, res143, freq, fsc):
    """Signed difference + local-agreement map for one pairwise-aligned pair."""
    va = ref_vol            # cryosparc (reference frame)
    vb = aligned_mov        # cryoDRGN, aligned onto cryosparc
    mask = solid_mask(va)
    diff = normalise_in_mask(vb, mask) - normalise_in_mask(va, mask)
    diff_rms = float(np.sqrt((diff.ravel()[mask.ravel()] ** 2).mean()))
    lcc = local_corr_map(va, vb, win)
    lcc_in = lcc[mask]
    lcc_med = float(np.nanmedian(lcc_in))
    lcc_lo = float(np.nanpercentile(lcc_in, 10))
    cc = cross_correlation(lowpass(vb, apix, 8.0), lowpass(va, apix, 8.0), mask)
    va_n = normalise_in_mask(va, mask)
    vb_n = normalise_in_mask(vb, mask)
    ssim = masked_ssim(vb_n, va_n, mask)
    nmi = masked_nmi(vb, va, mask)

    write_mrc(os.path.join(out, f"{tag}_localcorr.mrc"), np.nan_to_num(lcc), apix)

    mid = va.shape[0] // 2
    fig, ax = plt.subplots(1, 5, figsize=(20, 4.2))
    ax[0].imshow(va[mid], cmap="gray"); ax[0].set_title(f"{cs['method']}\n{cs['name']}")
    ax[1].imshow(vb[mid], cmap="gray")
    ax[1].set_title(f"{mover['method']}\n{mover['name']} (aligned)")
    dv = np.abs(diff[mask]).max() if mask.any() else 1.0
    ax[2].imshow(diff[mid], cmap="bwr", vmin=-dv, vmax=dv)
    ax[2].set_title(f"difference (cryoDRGN-cryoSPARC)\nRMS={diff_rms:.2f}")
    im = ax[3].imshow(np.where(mask[mid], lcc[mid], np.nan), cmap="RdYlGn",
                      vmin=0, vmax=1)
    ax[3].set_title(f"local agreement\nmedian={lcc_med:.2f}")
    fig.colorbar(im, ax=ax[3], fraction=0.046)
    ax[4].plot(freq, fsc); ax[4].axhline(0.5, color="gray", ls="--", lw=0.8)
    ax[4].axhline(0.143, color="r", ls=":", lw=0.8)
    ax[4].set_xlabel("1/A"); ax[4].set_ylabel("FSC"); ax[4].set_ylim(-0.1, 1.05)
    ax[4].set_title(f"FSC0.5={res05:.1f}A  FSC0.143={res143:.1f}A\nCC(lp)={cc:.2f}")
    for a_ in ax[:4]:
        a_.axis("off")
    fig.suptitle(tag, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(out, f"{tag}_pair.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"  [plot] {p}")
    return {"pair": tag, "cc": cc, "fsc05_A": res05, "fsc0143_A": res143,
            "diff_rms": diff_rms, "local_corr_median": lcc_med,
            "local_corr_p10": lcc_lo, "ssim": ssim, "nmi": nmi}


# --------------------------------------------------------------------------- #
def run_group(label, specs, args, out):
    print(f"\n=== {label} ===")
    maps, apix, phys = load_group(specs, args.work_box)
    if len(maps) < 2:
        print("  [skip] <2 maps")
        return None
    print(f"  {len(maps)} maps | working box {args.work_box} | apix {apix:.3f} "
          f"| phys {phys:.1f} A")
    cs = [m for m in maps if m["method"] == "cryosparc"]
    movers = [m for m in maps if m["method"] in ("cryodrgn", "subhetero")]
    if not cs:
        print("  [skip] no cryosparc reference maps")
        return None
    anchor = cs[0]
    lp = args.lowpass

    # --- montage: align every map to the anchor (display only) ---
    anchor["disp"] = anchor["vol"]
    for m in maps:
        if m is anchor:
            continue
        m["disp"] = compare_to_ref(anchor["vol"], m["vol"], apix,
                                   args.align_box, args.n_rot, lp)["aligned"]
    slice_montage(maps, out, label, anchor["name"])

    # --- cross-method grid: each mover pairwise-aligned onto each cryosparc ---
    cs_names = [m["name"] for m in cs]
    mv_names = [f"{m['method'][:4]}:{m['name']}" for m in movers]
    G_score = np.zeros((len(movers), len(cs)))
    G_r143 = np.zeros((len(movers), len(cs)))
    G_cc = np.zeros((len(movers), len(cs)))
    grid = {}
    print("  [grid] pairwise-aligning each cryoDRGN/subhetero map onto each "
          "CryoSPARC class ...")
    for i, mv in enumerate(movers):
        for j, c in enumerate(cs):
            res = compare_to_ref(c["vol"], mv["vol"], apix, args.align_box,
                                 args.n_rot, lp)
            grid[(i, j)] = res
            G_score[i, j] = res["score"]
            G_r143[i, j] = res["res143"]
            G_cc[i, j] = res["cc"]

    matrix_heatmap(G_cc, mv_names, cs_names, out, f"{label}_cc",
                   f"{label}: low-pass CC (cryoDRGN rows vs CryoSPARC cols, pairwise aligned)",
                   fmt="{:.2f}", vmin=0, vmax=1, cmap="viridis")
    fin = G_r143[np.isfinite(G_r143) & (G_r143 > 0)]
    vmax = float(np.percentile(fin, 95)) if fin.size else None
    matrix_heatmap(np.where(np.isfinite(G_r143), G_r143, np.nan), mv_names,
                   cs_names, out, f"{label}_fsc0143",
                   f"{label}: FSC0.143 res-of-agreement A (lower=more similar)",
                   fmt="{:.0f}", vmax=vmax, cmap="viridis_r", invert=True)

    # --- cryosparc vs cryosparc (are the classes even distinct?) ---
    CS = np.zeros((len(cs), len(cs)))
    for i in range(len(cs)):
        CS[i, i] = 1.0
        for j in range(i + 1, len(cs)):
            r = compare_to_ref(cs[i]["vol"], cs[j]["vol"], apix, args.align_box,
                               args.n_rot, lp)
            CS[i, j] = CS[j, i] = r["cc"]
    matrix_heatmap(CS, cs_names, cs_names, out, f"{label}_cs_vs_cs",
                   f"{label}: CryoSPARC class vs class low-pass CC (are classes distinct?)",
                   fmt="{:.2f}", vmin=0, vmax=1, cmap="viridis")

    # --- assignment + deep-dive ---
    assignments, pairs = [], []
    for i, mv in enumerate(movers):
        order = sorted(range(len(cs)), key=lambda j: -G_score[i, j])
        best, second = order[0], (order[1] if len(order) > 1 else order[0])
        assignments.append({
            "map": f"{mv['method']}:{mv['name']}",
            "best_match": cs[best]["name"],
            "score": float(G_score[i, best]),
            "cc": float(G_cc[i, best]),
            "fsc0143_A": float(G_r143[i, best]),
            "runner_up": cs[second]["name"],
            "runner_up_score": float(G_score[i, second]),
            "align_rot_deg": grid[(i, best)]["info"]["rot_deg"]})
        print(f"  [match] {mv['method']:9s} {mv['name']:9s} -> "
              f"{cs[best]['name']:4s} (score {G_score[i,best]:.2f}, "
              f"FSC0.143 {G_r143[i,best]:.1f}A, CC {G_cc[i,best]:.2f}); "
              f"next {cs[second]['name']} {G_score[i,second]:.2f}")
        res = grid[(i, best)]
        tag = f"{label}_{mv['name']}_vs_{cs[best]['name']}"
        pairs.append(pair_deepdive(mv, cs[best], res["aligned"], cs[best]["vol"],
                                   apix, args.local_win, out, tag,
                                   res["res05"], res["res143"], res["freq"],
                                   res["fsc"]))

    return {"label": label, "work_box": args.work_box, "apix": apix,
            "phys_A": phys,
            "cs_names": cs_names, "mover_names": mv_names,
            "cc_grid": G_cc.tolist(),
            "fsc0143_grid": np.where(np.isfinite(G_r143), G_r143, -1).tolist(),
            "score_grid": G_score.tolist(),
            "cs_vs_cs_cc": CS.tolist(),
            "assignments": assignments,
            "matched_pairs": pairs}


def write_summary(results, out):
    lines = ["# Map comparison: cryoDRGN basins vs CryoSPARC classes", "",
             "Every map reconstructed in its own pose frame was rigid-body "
             "aligned to a common reference before any metric. FSC crossings are "
             "the *resolution of agreement* between two maps (lower A = more "
             "similar). 'Local agreement' is a sliding-window correlation between "
             "a matched pair (green=agree, red=differ); true per-voxel local "
             "*resolution* needs half-maps, which were not provided.", ""]
    for r in results:
        if r is None:
            continue
        lines += [f"## {r['label']}", "",
                  f"- working box {r['work_box']}, apix {r['apix']:.3f} A, "
                  f"physical field {r['phys_A']:.1f} A",
                  "- each cryoDRGN/subhetero map was pairwise rigid-aligned onto "
                  "*each* CryoSPARC class; the match is the class with the best "
                  "FSC-based agreement score (not raw CC).", "",
                  "### Cross-method assignment (best CryoSPARC match)", "",
                  "| map | best match | score | CC(lp) | FSC0.143 (A) | rot (deg) "
                  "| runner-up (score) |",
                  "|---|---|---|---|---|---|---|"]
        for a in r["assignments"]:
            ru = f"{a['runner_up']} ({a['runner_up_score']:.2f})"
            lines.append(f"| {a['map']} | **{a['best_match']}** | {a['score']:.2f} "
                         f"| {a['cc']:.2f} | {a['fsc0143_A']:.1f} | "
                         f"{a['align_rot_deg']:.0f} | {ru} |")
        lines += ["", "### Matched-pair differences "
                  "(non-resolution structural metrics emphasised)", "",
                  "CC/SSIM/NMI/local-agree: higher = more similar; diff RMS: "
                  "lower = more similar. FSC columns kept for reference only.", "",
                  "| pair | CC | SSIM | NMI | local-agree median | local-agree p10 "
                  "| diff RMS | FSC0.143 (A) |",
                  "|---|---|---|---|---|---|---|---|"]
        for p in r["matched_pairs"]:
            lines.append(f"| {p['pair']} | {p['cc']:.3f} | {p['ssim']:.3f} | "
                         f"{p['nmi']:.3f} | {p['local_corr_median']:.2f} | "
                         f"{p['local_corr_p10']:.2f} | {p['diff_rms']:.2f} | "
                         f"{p['fsc0143_A']:.1f} |")
        lines.append("")
    p = os.path.join(out, "map_comparison_summary.md")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"[summary] {p}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", action="append", required=True,
                    help="GROUP:METHOD:NAME:path  (METHOD in cryodrgn|cryosparc|subhetero)")
    ap.add_argument("--work-box", type=int, default=192)
    ap.add_argument("--align-box", type=int, default=56)
    ap.add_argument("--n-rot", type=int, default=700)
    ap.add_argument("--local-win", type=int, default=15)
    ap.add_argument("--lowpass", type=float, default=8.0,
                    help="low-pass (A) for alignment/CC robustness")
    ap.add_argument("-o", "--out", default="results_cryodrgn/map_comparison")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    groups = {}
    for spec in args.map:
        parts = spec.split(":", 3)
        if len(parts) != 4:
            raise SystemExit(f"bad --map (need GROUP:METHOD:NAME:path): {spec}")
        g, method, name, path = parts
        groups.setdefault(g, []).append((method, name, path))

    results = []
    for g, specs in groups.items():
        results.append(run_group(g, specs, args, args.out))

    with open(os.path.join(args.out, "map_comparison_metrics.json"), "w",
              encoding="utf-8") as fh:
        json.dump([r for r in results if r], fh, indent=2)
    write_summary(results, args.out)
    print(f"\n[done] -> {args.out}")


if __name__ == "__main__":
    main()
