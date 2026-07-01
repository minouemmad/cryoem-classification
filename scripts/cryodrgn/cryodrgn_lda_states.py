#!/usr/bin/env python
"""LDA substate test: can a *supervised* discriminant axis recover the maps that
distinguish the "merged" CryoSPARC classes (J1497 P6/P10, P8/P9) that PCA and the
free-energy landscape wash out?

Motivation
----------
PCA / F(PC1) maximise *variance*, so a substate that differs by a 1%-variance
local event (a loop, a small domain rotation) is largely invisible - which is
why P6 and P10 looked identical (latent overlap 0.61, barrier 0.05 kT, map CC
0.91).  LDA instead maximises *between-class* over *within-class* scatter, so if
any subtle but systematic feature separates two classes, the LDA axis points at
it even when it carries little total variance.

The decisive test is NOT the LDA number itself but what the axis *decodes to*:
  1. find, for each target pair, the LDA direction that best separates them;
  2. decode volumes at the two extremes of that axis (the LDA-sharpened
     endpoints) plus a morph between them;
  3. difference-map the endpoints, compute FSC + real-space CC;
  4. SPLIT-HALF reproducibility: redo the endpoints on two independent random
     halves of the particles and correlate the two difference maps.

Read-out:
  * endpoints decode to *different* density (CC notably < the medoid CC of
    decode_states) AND the difference map reproduces across halves (high
    diff-map CC)  =>  P10 is a real, reproducible local substate of P6.
  * endpoints stay ~identical (CC ~0.9), difference map is noise and does NOT
    reproduce across halves  =>  the split is not backed by recoverable density;
    stop chasing five states.

Run with the cryoDRGN env from the repo root::

    python scripts/cryodrgn/cryodrgn_lda_states.py \
      --dataset "J1497:results_cryodrgn/J1497_real/train/z.100.pkl:data/gP25W6J1497_passthrough_particles_all_classes.cs:data/cryosparc_P25_J1497_00000_particles.cs:6,7,8,9,10:results_cryodrgn/J1497_real/train/weights.100.pkl:results_cryodrgn/J1497_real/train/config.yaml:data/J1497_classes" \
      --pairs 6-10,8-9 --n-traj 6 -d 64 --apix 4.15 --run \
      -o results_cryodrgn/lda_states

For the J1442 3-class control add another --dataset with --pairs 6-7 (its only
"merged" pair).  Decode at -d 64 first (fast, ~minutes on CPU); if a real
difference appears, re-decode the two endpoints at full box (-d 0) for FSC at
higher resolution and hand the maps to a local-resolution tool (blocres/ResMap)
- per-voxel local resolution is out of scope for a pure-numpy script.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for p in (_REPO, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import cryodrgn_latent_gmm as clg
from cryodrgn_decode_states import (cross_correlation, eval_vol_cmd,
                                    load_official_membership, read_mrc,
                                    representative_z)


# --------------------------------------------------------------------------- #
def conditional_mean_z(z_a, axis_val, lo, hi, n_pts, min_n=200):
    """On-manifold latent points: mean z of particles in each window of an axis.

    Walks `axis_val` from `lo` to `hi` in `n_pts` bins; if a bin is too sparse it
    falls back to the `min_n` particles nearest the bin centre, so every decoded
    point sits on the populated manifold instead of extrapolating off it.
    """
    edges = np.linspace(lo, hi, n_pts + 1)
    pts = []
    for i in range(n_pts):
        m = (axis_val >= edges[i]) & (axis_val <= edges[i + 1])
        if m.sum() < 10:
            c = 0.5 * (edges[i] + edges[i + 1])
            idx = np.argsort(np.abs(axis_val - c))[:min_n]
            pts.append(z_a[idx].mean(0))
        else:
            pts.append(z_a[m].mean(0))
    return np.asarray(pts, dtype=np.float32)


def fsc_curve(v1, v2, apix):
    """Fourier shell correlation; returns (freq_inv_A, fsc, res_half_A)."""
    f1 = np.fft.fftshift(np.fft.fftn(v1))
    f2 = np.fft.fftshift(np.fft.fftn(v2))
    n = v1.shape[0]
    c = n // 2
    zz, yy, xx = np.indices(v1.shape)
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2 + (zz - c) ** 2).astype(int)
    r = r.ravel()
    nb = c
    num = np.bincount(r, (f1 * np.conj(f2)).real.ravel(), minlength=nb + 1)
    p1 = np.bincount(r, (np.abs(f1) ** 2).ravel(), minlength=nb + 1)
    p2 = np.bincount(r, (np.abs(f2) ** 2).ravel(), minlength=nb + 1)
    denom = np.sqrt(p1 * p2)
    fsc = num[:nb] / np.clip(denom[:nb], 1e-12, None)
    freq = np.arange(nb) / (n * apix)        # cycles / Angstrom
    res_half = np.inf
    below = np.where(fsc < 0.5)[0]
    if below.size and below[0] > 0:
        res_half = 1.0 / freq[below[0]]
    return freq, fsc, float(res_half)


def normalise_in_mask(v, mask):
    m = mask.ravel().astype(bool)
    vv = v.ravel().astype(np.float64)
    mu, sd = vv[m].mean(), vv[m].std()
    return ((vv - mu) / (sd if sd > 0 else 1.0)).reshape(v.shape)


def best_lda_axis(L, lab, a, b):
    """Index of the LDA component that maximally separates classes a vs b
    (Fisher ratio on that 1-D axis), plus the per-axis separations."""
    seps = []
    for k in range(L.shape[1]):
        la, lb = L[lab == a, k], L[lab == b, k]
        pooled = np.sqrt(0.5 * (la.var() + lb.var())) + 1e-12
        seps.append(abs(la.mean() - lb.mean()) / pooled)
    seps = np.asarray(seps)
    return int(np.argmax(seps)), seps


# --------------------------------------------------------------------------- #
def analyse(label, z_path, pass_cs, cs, protein_idx, weights, config, class_dir,
            n_dummies, pairs, n_traj, downsample, apix, out, run, seed):
    print(f"\n=== {label} ===")
    names = [f"P{p}" for p in protein_idx]
    idx_of = {p: j for j, p in enumerate(protein_idx)}
    k = len(names)
    ds_dir = os.path.join(out, label)
    os.makedirs(ds_dir, exist_ok=True)

    z = clg.load_latent(z_path)
    z_a, cryo_post, cryo_hard, uid_a, n_prot = clg.align_z_to_posteriors(
        z, pass_cs, cs, n_dummies, protein_idx)
    member = load_official_membership(class_dir, protein_idx)
    if member is not None:
        official = np.array([member.get(int(u), -1) for u in uid_a.tolist()])
        ok = official >= 0
        if ok.any():
            agree = float(np.mean(official[ok] == cryo_hard[ok]))
            print(f"[member] {ok.sum():,}/{len(official):,} matched; "
                  f"argmax vs official agreement {agree*100:.1f}%")
            cryo_hard = np.where(ok, official, cryo_hard)

    # ---- supervised LDA on standardized latent --------------------------------
    scaler = StandardScaler().fit(z_a)
    Xs = scaler.transform(z_a)
    lda = LinearDiscriminantAnalysis(n_components=k - 1, solver="svd")
    L = lda.fit_transform(Xs, cryo_hard)
    # quick 5-fold balanced accuracy for context
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(Xs, cryo_hard):
        m = LinearDiscriminantAnalysis(solver="svd").fit(Xs[tr], cryo_hard[tr])
        pred = m.predict(Xs[te])
        recs = [np.mean(pred[cryo_hard[te] == c] == c)
                for c in range(k) if np.any(cryo_hard[te] == c)]
        accs.append(np.mean(recs))
    print(f"[lda] {k - 1} discriminant axes | 5-fold balanced acc "
          f"{np.mean(accs):.3f} +/- {np.std(accs):.3f}")

    medoid_z = representative_z(z_a, cryo_hard, k, "medoid")

    # ---- two independent split-half LDA models (rigorous reproducibility) -----
    # Each half re-fits StandardScaler + LDA from scratch over ALL classes, so
    # the reproducibility test asks whether an *independently trained*
    # discriminant rediscovers the SAME structural difference (not merely that a
    # conditional mean is stable).
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(z_a))
    half_id = np.zeros(len(z_a), int)
    half_id[order[1::2]] = 1
    half_models = []
    for h in range(2):
        hm = half_id == h
        sc = StandardScaler().fit(z_a[hm])
        ld = LinearDiscriminantAnalysis(n_components=k - 1, solver="svd").fit(
            sc.transform(z_a[hm]), cryo_hard[hm])
        half_models.append((hm, sc, ld))

    # ---- assemble all z to decode (one eval_vol call) -------------------------
    rows = []                       # (name, z)
    plan = []                       # per-pair bookkeeping for analysis
    for (pa, pb) in pairs:
        if pa not in idx_of or pb not in idx_of:
            print(f"[pair] skip {pa}-{pb}: not in {label}")
            continue
        a, b = idx_of[pa], idx_of[pb]
        kbest, seps = best_lda_axis(L, cryo_hard, a, b)
        Lk = L[:, kbest]
        sel = (cryo_hard == a) | (cryo_hard == b)
        # orient so class a is the low end
        if Lk[cryo_hard == a].mean() > Lk[cryo_hard == b].mean():
            Lk = -Lk
        lo = np.percentile(Lk[sel], 5)
        hi = np.percentile(Lk[sel], 95)
        traj = conditional_mean_z(z_a, Lk, lo, hi, max(n_traj, 2))
        z_lo, z_hi = traj[0], traj[-1]

        # split-half endpoints from independently re-fit LDA models
        halves = []
        for (hm, sc, ld) in half_models:
            ch = cryo_hard[hm]
            Lh = ld.transform(sc.transform(z_a[hm]))
            kb, _ = best_lda_axis(Lh, ch, a, b)
            Lhk = Lh[:, kb]
            if Lhk[ch == a].mean() > Lhk[ch == b].mean():
                Lhk = -Lhk
            selab = (ch == a) | (ch == b)
            loh = np.percentile(Lhk[selab], 5)
            hih = np.percentile(Lhk[selab], 95)
            th = conditional_mean_z(z_a[hm], Lhk, loh, hih, 2)
            halves.append((th[0], th[-1]))

        pfx = f"{pa}_{pb}"
        base = len(rows)
        rows.append((f"{pfx}_lo", z_lo))
        rows.append((f"{pfx}_hi", z_hi))
        for t in range(len(traj)):
            rows.append((f"{pfx}_traj{t:02d}", traj[t]))
        rows.append((f"{pfx}_medoidA", medoid_z[a]))
        rows.append((f"{pfx}_medoidB", medoid_z[b]))
        for h, (lz, hz) in enumerate(halves):
            rows.append((f"{pfx}_h{h}lo", lz))
            rows.append((f"{pfx}_h{h}hi", hz))
        plan.append({"pa": pa, "pb": pb, "a": a, "b": b, "axis": kbest,
                     "sep_best": float(seps[kbest]),
                     "sep_all": seps.tolist(), "row_base": base,
                     "n_traj": len(traj),
                     "Lk_a": Lk[cryo_hard == a], "Lk_b": Lk[cryo_hard == b],
                     "na": int((cryo_hard == a).sum()),
                     "nb": int((cryo_hard == b).sum())})

    if not plan:
        print(f"[{label}] no valid pairs; nothing to do")
        return None

    row_labels = [r[0] for r in rows]
    zmat = np.asarray([r[1] for r in rows], dtype=np.float32)
    zfile = os.path.join(ds_dir, f"{label}_lda_zfile.txt")
    np.savetxt(zfile, zmat, fmt="%.6f")
    with open(os.path.join(ds_dir, f"{label}_lda_labels.txt"), "w") as fh:
        fh.write("\n".join(f"{i}\t{lab}" for i, lab in enumerate(row_labels)))
    cmd = eval_vol_cmd(weights, config, zfile, ds_dir, downsample, apix,
                       prefix=f"{label}_lda")
    print(f"[zfile] {len(rows)} latent points -> {zfile}")
    print("[eval_vol]", " ".join(cmd))
    if run:
        if not (os.path.exists(weights) and os.path.exists(config)):
            print(f"[eval_vol] SKIP - weights/config missing")
        else:
            print(f"[eval_vol] decoding {len(rows)} volumes (box "
                  f"{downsample or 'full'})...")
            subprocess.run(cmd, check=True)

    # ---- read volumes back (1-based vol index = row index + 1) ----------------
    def vol(i):
        p = os.path.join(ds_dir, f"{label}_lda{i + 1:03d}.mrc")
        return read_mrc(p) if os.path.exists(p) else None

    results = []
    for pl in plan:
        base = pl["row_base"]
        v_lo = vol(base + 0)
        v_hi = vol(base + 1)
        nt = pl["n_traj"]
        mA = vol(base + 2 + nt)
        mB = vol(base + 2 + nt + 1)
        h0lo = vol(base + 2 + nt + 2)
        h0hi = vol(base + 2 + nt + 3)
        h1lo = vol(base + 2 + nt + 4)
        h1hi = vol(base + 2 + nt + 5)
        pa, pb = pl["pa"], pl["pb"]
        entry = {"pair": f"P{pa}-P{pb}", "lda_axis": pl["axis"],
                 "fisher_sep": pl["sep_best"], "n_a": pl["na"], "n_b": pl["nb"]}
        if v_lo is None or v_hi is None:
            print(f"[P{pa}-P{pb}] volumes missing - run with --run")
            results.append(entry | {"status": "no_volumes"})
            continue

        ref = np.mean([v for v in (v_lo, v_hi, mA, mB) if v is not None], axis=0)
        mask = ref > (ref.mean() + 0.5 * ref.std())

        cc_endpoints = cross_correlation(v_lo, v_hi, mask)
        cc_medoid = (cross_correlation(mA, mB, mask)
                     if mA is not None and mB is not None else float("nan"))
        freq, fsc, res_half = fsc_curve(v_lo, v_hi, apix)

        # difference map (mask-normalised) + reproducibility across halves
        d_full = normalise_in_mask(v_hi, mask) - normalise_in_mask(v_lo, mask)
        diff_rms = float(np.sqrt((d_full.ravel()[mask.ravel()] ** 2).mean()))
        repro = float("nan")
        if all(v is not None for v in (h0lo, h0hi, h1lo, h1hi)):
            d0 = normalise_in_mask(h0hi, mask) - normalise_in_mask(h0lo, mask)
            d1 = normalise_in_mask(h1hi, mask) - normalise_in_mask(h1lo, mask)
            repro = cross_correlation(d0, d1, mask)

        print(f"[P{pa}-P{pb}] LDA axis {pl['axis']} Fisher sep {pl['sep_best']:.2f}"
              f" | endpoint CC {cc_endpoints:.3f} (medoid CC {cc_medoid:.3f})"
              f" | diff RMS {diff_rms:.3f} | FSC0.5 {res_half:.1f} A"
              f" | split-half diff-map CC {repro:.3f}")
        entry.update({"status": "ok", "cc_endpoints": cc_endpoints,
                      "cc_medoid": cc_medoid, "diff_rms": diff_rms,
                      "fsc05_resolution_A": res_half,
                      "splithalf_diffmap_cc": repro,
                      "_v_lo": v_lo, "_v_hi": v_hi, "_diff": d_full,
                      "_mask": mask, "_freq": freq, "_fsc": fsc})
        results.append(entry)

    plot_dataset(label, plan, results, apix, ds_dir)
    clean = [{k2: v2 for k2, v2 in r.items() if not k2.startswith("_")}
             for r in results]
    return {"label": label, "balanced_acc": float(np.mean(accs)),
            "pairs": clean}


# --------------------------------------------------------------------------- #
def plot_dataset(label, plan, results, apix, ds_dir):
    ok = [r for r in results if r.get("status") == "ok"]
    if not ok:
        return
    npairs = len(ok)
    fig, axes = plt.subplots(npairs, 4, figsize=(16, 4.0 * npairs),
                             squeeze=False)
    for row, (pl, r) in enumerate(zip(plan, ok)):
        ax = axes[row][0]
        ax.hist(pl["Lk_a"], bins=60, alpha=0.6, color="tab:blue",
                label=f"P{pl['pa']}", density=True)
        ax.hist(pl["Lk_b"], bins=60, alpha=0.6, color="tab:red",
                label=f"P{pl['pb']}", density=True)
        ax.set_title(f"{r['pair']}  LDA axis {r['lda_axis']}\nFisher sep "
                     f"{r['fisher_sep']:.2f}")
        ax.set_xlabel("LDA score"); ax.legend(fontsize=8)

        mid = r["_v_lo"].shape[0] // 2
        for col, (vol_img, ttl) in enumerate(
                [(r["_v_lo"], f"P{pl['pa']} end (lo)"),
                 (r["_v_hi"], f"P{pl['pb']} end (hi)")], start=1):
            axc = axes[row][col]
            axc.imshow(vol_img[mid], cmap="gray")
            axc.set_title(ttl); axc.axis("off")

        axd = axes[row][3]
        d = r["_diff"][mid]
        vmax = np.abs(r["_diff"]).max() or 1.0
        axd.imshow(d, cmap="bwr", vmin=-vmax, vmax=vmax)
        axd.set_title(f"diff (hi-lo)\nendpt CC {r['cc_endpoints']:.2f} | "
                      f"split-half {r['splithalf_diffmap_cc']:.2f}")
        axd.axis("off")
    fig.suptitle(f"{label}: LDA-sharpened substate test "
                 f"(apix {apix} A/px, box {results[0]['_v_lo'].shape[0]})",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    p = os.path.join(ds_dir, f"{label}_lda_substates.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[plot] {p}")

    # FSC overlay
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.8))
    for r in ok:
        ax2.plot(r["_freq"], r["_fsc"], label=f"{r['pair']} "
                 f"(FSC0.5 {r['fsc05_resolution_A']:.1f} A)")
    ax2.axhline(0.5, color="k", ls="--", lw=0.8)
    ax2.axhline(0.143, color="gray", ls=":", lw=0.8)
    ax2.set_xlabel("spatial frequency (1/A)")
    ax2.set_ylabel("FSC (lo vs hi endpoint)")
    ax2.set_title(f"{label}: endpoint FSC")
    ax2.legend(fontsize=8); ax2.set_ylim(-0.1, 1.05)
    fig2.tight_layout()
    p2 = os.path.join(ds_dir, f"{label}_lda_fsc.png")
    fig2.savefig(p2, dpi=150)
    plt.close(fig2)
    print(f"[plot] {p2}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", action="append", required=True,
                    help="LABEL:Z:PASS_CS:CS:PROT_IDX:WEIGHTS:CONFIG[:CLASS_DIR]")
    ap.add_argument("--pairs", default="6-10,8-9",
                    help="comma list of protein-idx pairs, e.g. 6-10,8-9")
    ap.add_argument("--n-dummies", type=int, default=6)
    ap.add_argument("--n-traj", type=int, default=6,
                    help="morph points along the LDA axis (incl. endpoints)")
    ap.add_argument("-d", "--downsample", type=int, default=64,
                    help="decode box (default 64; 0=full for higher-res FSC)")
    ap.add_argument("--apix", type=float, default=4.15,
                    help="A/px of decoded box (box64 of D128@2.075 ~ 4.15)")
    ap.add_argument("--run", action="store_true",
                    help="actually call cryodrgn eval_vol")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--out", default="results_cryodrgn/lda_states")
    args = ap.parse_args()

    pairs = []
    for tok in args.pairs.split(","):
        tok = tok.strip()
        if not tok:
            continue
        x, y = tok.split("-")
        pairs.append((int(x), int(y)))

    os.makedirs(args.out, exist_ok=True)
    res = []
    for spec in args.dataset:
        parts = spec.split(":")
        if len(parts) < 7:
            raise SystemExit(f"--dataset needs >=7 fields, got: {spec}")
        label, z_path, pass_cs, cs, prot, weights, config = parts[:7]
        class_dir = parts[7] if len(parts) > 7 else None
        protein_idx = [int(x) for x in prot.split(",")]
        r = analyse(label, z_path, pass_cs, cs, protein_idx, weights, config,
                    class_dir, args.n_dummies, pairs, args.n_traj,
                    args.downsample, args.apix, args.out, args.run, args.seed)
        if r:
            res.append(r)

    with open(os.path.join(args.out, "lda_states_metrics.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\n[done] LDA substate test -> {args.out}")


if __name__ == "__main__":
    main()
