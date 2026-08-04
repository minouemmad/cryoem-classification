#!/usr/bin/env python
"""Hierarchical uncertainty: substates *inside* free-energy basins.

The project's central idea is that a "state" may be hierarchical:

    particles -> cryoDRGN -> ENERGETIC BASINS      (Stage 1, unsupervised)
                              |
                              +-- within a basin -> STRUCTURAL SUBSTATES  (Stage 2)

Stage 1 (energetic basins) is handled by ``cryodrgn_free_energy.py`` /
``cryodrgn_basin_occupancy.py``.  This script answers Stage 2 *without* going
back to CryoSPARC labels, using two model-free tests, plus it quantifies the two
levels of uncertainty the PI cares about:

  LEVEL 1  uncertainty in basin populations
           -> bootstrap of the fraction of particles in each basin.

  LEVEL 2  uncertainty in substate assignments *inside* a basin
           -> Option 3 (local density peaks) with split-half reproducibility.

Option 3 -- local density peaks (the main Stage-2 test)
    For each populated basin, restrict to its particles, take the dominant
    *within-basin* latent axis (local PC1), and build the 1-D free energy
    F(local PC1) = -log p.  Sub-wells separated by a barrier >~ ``--sub-barrier``
    kT are candidate substates.  A split-half control refits the whole thing on
    two disjoint halves: a substate is only believed if BOTH halves recover the
    same number of sub-wells and learn the *same* axis (|cos| high).  This is the
    unsupervised complement to the (supervised) LDA test already done.

Option 4 -- diffusion map (slow-coordinate cross-check)
    A diffusion map is built on a subsample of the full latent.  Its eigenvalue
    spectrum has a *spectral gap* after k coordinates when the data has k
    metastable states; the diffusion coordinates (DC1, DC2) are the slowest
    coordinates and sometimes separate substates that PCA smears.  We report the
    spectral gap (an independent basin count) and whether DC1 is just PC1
    (corr) or reveals extra structure.

Options intentionally NOT implemented:
  * Option 1 (LDA) is supervised on CryoSPARC labels -> already done separately
    (cryodrgn_lda_states.py); kept as supporting evidence, not a primary test.
  * Option 2 (focused hetero-refine) is the CryoSPARC image-space confirmation
    step, driven by the plan this script emits -- not a local computation.
  * Option 5 (Markov-state models) needs *observed transitions* / time-ordering.
    Cryo-EM particles are independent snapshots with no time axis, so there are
    no transitions to estimate -- MSMs are undefined here, not merely overkill.

Run with the cryoDRGN env from repo root::

    python scripts/cryodrgn/cryodrgn_within_basin_substates.py \
      --dataset "J1442:results_cryodrgn/J1442_gP25_WT_POSE_BIAS/train_z10/z.100.pkl:data/cryosparc_P25_J1442_passthrough_particles_all_classes.cs:data/cryosparc_P25_J1442_00000_particles.cs:6,7,8" \
      --dataset "J1497:results_cryodrgn/J1497_real/train/z.100.pkl:data/gP25W6J1497_passthrough_particles_all_classes.cs:data/cryosparc_P25_J1497_00000_particles.cs:6,7,8,9,10" \
      --dataset "J264:results_cryodrgn/J264_real/pilot_z10/z.49.pkl:data/J264/cryosparc_P7_J264_passthrough_particles_all_classes_blob.cs:data/J264/cryosparc_P7_J264_00062_particles_alignments3D_multi.cs:6,7,8,9,10,11,12,13,14" \
      --n-dummies 6 --barrier-kt 0.5 --sub-barrier 0.5 \
      -o results_cryodrgn/within_basin_substates
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)  # scripts/ holds gmm_pipeline
_REPO = os.path.dirname(_SCRIPTS)
for p in (_REPO, _SCRIPTS, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import cryodrgn_latent_gmm as clg
import cryodrgn_basin_occupancy as cbo
import cryodrgn_free_energy as fe


# --------------------------------------------------------------------------- #
# Basin assignment (mirrors cryodrgn_basin_occupancy.analyse, but also keeps the
# full standardised latent so we can look *inside* each basin)
# --------------------------------------------------------------------------- #
def assign_basins(Xs, cryo_hard, protein_idx, barrier_kt, min_pop,
                  nx, ny, bw_scale, sub, seed):
    pca = PCA(n_components=2, random_state=seed).fit(Xs)
    scores = pca.transform(Xs)
    pc1, pc2 = scores[:, 0], scores[:, 1]
    cmeans = [pc1[cryo_hard == j].mean() for j in range(len(protein_idx))]
    if np.polyfit(range(len(cmeans)), cmeans, 1)[0] < 0:
        pc1 = -pc1
    F, gx, gy = cbo.free_energy_2d(pc1, pc2, nx, ny, bw_scale, sub, seed)
    lab2, centers = cbo._flood(F, barrier_kt)
    part = cbo.assign_particles(pc1, pc2, gx, gy, lab2)
    min_count = int(min_pop * len(pc1))
    lab2, centers, part, n_basin = cbo.merge_minor_basins(
        lab2, centers, gx, gy, part, min_count)
    order = sorted(range(n_basin), key=lambda b: gx[centers[b][1]])
    relabel = {old: new for new, old in enumerate(order)}
    part = np.array([relabel[b] for b in part])
    return part, n_basin, np.column_stack([pc1, pc2])


def basin_population_uncertainty(part, n_basin, n_boot, seed):
    """LEVEL 1: bootstrap the basin-population fractions."""
    rng = np.random.default_rng(seed)
    n = len(part)
    pops = np.zeros((n_boot, n_basin))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        pops[b] = np.bincount(part[idx], minlength=n_basin) / n
    mean = np.bincount(part, minlength=n_basin) / n
    return mean, pops.mean(0), pops.std(0)


# --------------------------------------------------------------------------- #
# Option 3: local density peaks inside a basin (+ split-half reproducibility)
# --------------------------------------------------------------------------- #
def _wells_along_axis(vals, sub_barrier, grid_n=200, bw_scale=1.0):
    grid = np.linspace(vals.min(), vals.max(), grid_n)
    F, _ = fe.free_energy_1d(vals, grid, bw_scale=bw_scale)
    minima, barriers = fe.basin_analysis(F, grid, barrier_kt=sub_barrier)
    return grid, F, minima, barriers


def within_basin_substates(Xs_basin, sub_barrier, seed, min_half=400,
                           bw_scale=1.0):
    """Option 3 for one basin.  Returns a dict describing sub-well structure and
    split-half reproducibility of the dominant within-basin axis."""
    n = len(Xs_basin)
    # local axis = 1st PC computed *inside* the basin
    pca = PCA(n_components=min(5, Xs_basin.shape[1]),
              random_state=seed).fit(Xs_basin)
    axis = pca.components_[0]
    lpc1 = Xs_basin @ axis
    grid, F, minima, barriers = _wells_along_axis(lpc1, sub_barrier,
                                                  bw_scale=bw_scale)
    n_sub = int(len(minima))

    # substate label per particle = nearest sub-well along local PC1
    if n_sub >= 2:
        well_pos = grid[minima]
        sub_label = np.argmin(np.abs(lpc1[:, None] - well_pos[None, :]), axis=1)
    else:
        sub_label = np.zeros(n, dtype=int)

    # --- split-half reproducibility: refit PCA independently on 2 halves ---
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    ha, hb = perm[: n // 2], perm[n // 2:]
    repro = {"axis_cos": None, "n_sub_a": None, "n_sub_b": None,
             "curve_corr": None, "reproducible": False}
    if min(len(ha), len(hb)) >= min_half:
        pa = PCA(n_components=1, random_state=seed).fit(Xs_basin[ha])
        pb = PCA(n_components=1, random_state=seed + 1).fit(Xs_basin[hb])
        ax_a, ax_b = pa.components_[0], pb.components_[0]
        cos = float(abs(ax_a @ ax_b))
        _, _, min_a, _ = _wells_along_axis(Xs_basin[ha] @ ax_a, sub_barrier,
                                           bw_scale=bw_scale)
        _, _, min_b, _ = _wells_along_axis(Xs_basin[hb] @ ax_b, sub_barrier,
                                           bw_scale=bw_scale)
        # correlate the two half-curves on a shared oriented grid
        sgn = 1.0 if cos >= 0 else -1.0
        va = Xs_basin[ha] @ ax_a
        vb = sgn * (Xs_basin[hb] @ ax_b)
        lo = min(va.min(), vb.min())
        hi = max(va.max(), vb.max())
        g = np.linspace(lo, hi, 200)
        Fa, _ = fe.free_energy_1d(va, g, bw_scale=bw_scale)
        Fb, _ = fe.free_energy_1d(vb, g, bw_scale=bw_scale)
        curve_corr = float(np.corrcoef(Fa, Fb)[0, 1])
        repro = {
            "axis_cos": cos,
            "n_sub_a": int(len(min_a)),
            "n_sub_b": int(len(min_b)),
            "curve_corr": curve_corr,
            # believable substate split: both halves agree on >=2 wells,
            # they learn the same axis, and the profiles are highly correlated
            "reproducible": bool(n_sub >= 2 and len(min_a) >= 2
                                 and len(min_b) >= 2 and cos >= 0.9
                                 and curve_corr >= 0.9),
        }

    deepest_barrier = (max(b["depth_from_shallower_well_kt"]
                           for b in barriers) if barriers else 0.0)
    return {
        "n_particles": int(n),
        "n_sub_wells": n_sub,
        "sub_well_positions": grid[minima].tolist() if n_sub else [],
        "deepest_sub_barrier_kt": float(deepest_barrier),
        "sub_barriers": barriers,
        "reproducibility": repro,
        "_grid": grid, "_F": F, "_lpc1": lpc1, "_sub_label": sub_label,
    }


# --------------------------------------------------------------------------- #
# Option 4: diffusion map (slow coordinates + spectral gap)
# --------------------------------------------------------------------------- #
def diffusion_map(Xs, n_sample, seed, n_evec=8, eps_scale=1.0):
    """Anisotropic (alpha=1) diffusion map on a random subsample.

    Returns diffusion coordinates (sample x n_evec), eigenvalues, and the index
    of the subsample rows so we can colour by class/basin."""
    rng = np.random.default_rng(seed)
    n = len(Xs)
    take = min(n_sample, n)
    idx = rng.choice(n, size=take, replace=False)
    X = Xs[idx]
    # pairwise squared distances
    sq = np.sum(X * X, axis=1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(D2, 0, out=D2)
    # kernel bandwidth = median nonzero distance (robust), scaled
    med = np.median(D2[D2 > 0])
    eps = eps_scale * med
    K = np.exp(-D2 / eps)
    # alpha=1 normalisation removes sampling-density bias -> approximates the
    # Laplace-Beltrami operator (geometry, not density)
    q = K.sum(1)
    K = K / (q[:, None] * q[None, :])
    d = K.sum(1)
    # symmetric normalised transition matrix (same spectrum as row-stochastic)
    dinv = 1.0 / np.sqrt(d)
    Ms = (dinv[:, None] * K) * dinv[None, :]
    Ms = 0.5 * (Ms + Ms.T)
    evals, evecs = np.linalg.eigh(Ms)
    order = np.argsort(evals)[::-1]
    evals = evals[order][: n_evec + 1]
    evecs = evecs[:, order][:, : n_evec + 1]
    # back to diffusion-map eigenvectors of the Markov matrix
    psi = dinv[:, None] * evecs
    # drop the trivial first (constant) eigenvector
    dc = psi[:, 1:] * evals[1:][None, :]
    return {"idx": idx, "dc": dc, "evals": evals}


def spectral_gap(evals):
    """Largest relative drop between consecutive non-trivial eigenvalues ->
    suggested number of metastable states (index of the gap)."""
    ev = evals[1:]                      # drop trivial lambda_0 = 1
    ev = ev[ev > 1e-9]
    if len(ev) < 2:
        return 1, []
    gaps = ev[:-1] - ev[1:]
    k = int(np.argmax(gaps)) + 1        # states = position of biggest gap
    return k, gaps.tolist()


# --------------------------------------------------------------------------- #
def analyse(label, z_path, pass_cs, cs, protein_idx, n_dummies, args):
    print(f"\n=== {label} ===")
    z = clg.load_latent(z_path)
    z_a, cryo_post, cryo_hard, uid_a, n_prot = clg.align_z_to_posteriors(
        z, pass_cs, cs, n_dummies, protein_idx)
    Xs = StandardScaler().fit_transform(z_a)

    part, n_basin, scores = assign_basins(
        Xs, cryo_hard, protein_idx, args.barrier_kt, args.min_pop,
        args.grid, args.grid, args.bw_scale, args.sub, args.seed)
    pmean, pboot, pstd = basin_population_uncertainty(
        part, n_basin, args.n_boot, args.seed)
    print(f"[{label}] {n_basin} basin(s); populations "
          + ", ".join(f"B{b+1}={pmean[b]:.3f}+/-{pstd[b]:.3f}"
                      for b in range(n_basin)))

    # ---- Option 3: within-basin substates ----
    basins = []
    for b in range(n_basin):
        sel = np.where(part == b)[0]
        if len(sel) < args.min_basin:
            print(f"[{label}] basin {b+1}: {len(sel)} particles < "
                  f"{args.min_basin}, skipped")
            continue
        info = within_basin_substates(Xs[sel], args.sub_barrier, args.seed,
                                      bw_scale=args.bw_scale)
        info["basin"] = b + 1
        info["population"] = float(pmean[b])
        info["population_std"] = float(pstd[b])
        # cross-tab sub-well vs CryoSPARC class
        ct = np.zeros((info["n_sub_wells"] if info["n_sub_wells"] else 1,
                       len(protein_idx)), dtype=int)
        for s, c in zip(info["_sub_label"], cryo_hard[sel]):
            ct[s, c] += 1
        info["subwell_class_counts"] = ct.tolist()
        info["_sel"] = sel
        basins.append(info)
        rep = info["reproducibility"]
        print(f"[{label}] basin {b+1}: {len(sel):,} part, "
              f"{info['n_sub_wells']} sub-well(s), deepest sub-barrier "
              f"{info['deepest_sub_barrier_kt']:.2f} kT, "
              f"split-half repro={rep['reproducible']} "
              f"(cos={rep['axis_cos']}, corr={rep['curve_corr']})")

    # ---- Option 4: diffusion map ----
    dm = diffusion_map(Xs, args.dm_sample, args.seed, eps_scale=args.eps_scale)
    kgap, gaps = spectral_gap(dm["evals"])
    dc1 = dm["dc"][:, 0]
    pc1_sub = scores[dm["idx"], 0]
    dc_pc1_corr = float(abs(np.corrcoef(dc1, pc1_sub)[0, 1]))
    print(f"[{label}] diffusion map: spectral-gap suggests ~{kgap} metastable "
          f"state(s); |corr(DC1, PC1)| = {dc_pc1_corr:.2f}")

    return {
        "label": label,
        "n_particles": int(len(part)),
        "zdim": int(z_a.shape[1]),
        "protein_idx": list(protein_idx),
        "n_basins": int(n_basin),
        "basin_populations": pmean.tolist(),
        "basin_population_std": pstd.tolist(),
        "substates": [{k: v for k, v in b.items() if not k.startswith("_")}
                      for b in basins],
        "diffusion_spectral_gap_k": int(kgap),
        "diffusion_eigenvalues": dm["evals"].tolist(),
        "diffusion_dc1_pc1_corr": dc_pc1_corr,
        "_scores": scores, "_part": part, "_cryo_hard": cryo_hard,
        "_basins": basins, "_dm": dm, "_kgap": kgap,
    }


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_dataset(res, outdir):
    label = res["label"]
    basins = res["_basins"]
    protein_idx = res["protein_idx"]
    n_basin = res["n_basins"]
    class_names = [f"P{j}" for j in protein_idx]

    ncol = max(len(basins), 1)
    fig, axes = plt.subplots(2, ncol, figsize=(5 * ncol, 9), squeeze=False)

    # row 0: within-basin F(local PC1) with sub-wells
    for c, info in enumerate(basins):
        ax = axes[0][c]
        ax.plot(info["_grid"], info["_F"], color="navy", lw=2)
        for m in np.argmin(np.abs(info["_grid"][:, None]
                                  - np.array(info["sub_well_positions"])[None, :]),
                           axis=0) if info["sub_well_positions"] else []:
            ax.plot(info["_grid"][m], info["_F"][m], "v", color="crimson", ms=11)
        rep = info["reproducibility"]
        ok = "REPRODUCIBLE" if rep["reproducible"] else "not reproducible"
        ax.set_title(f"Basin {info['basin']}  (pop {info['population']:.2f})\n"
                     f"{info['n_sub_wells']} sub-well(s), "
                     f"barrier {info['deepest_sub_barrier_kt']:.2f} kT\n{ok}",
                     fontsize=10)
        ax.set_xlabel("within-basin local PC1")
        ax.set_ylabel("F = -log p  (kT)")
    for c in range(len(basins), ncol):
        axes[0][c].axis("off")

    # row 1 col 0: latent plane coloured by basin
    ax = axes[1][0]
    scores, part = res["_scores"], res["_part"]
    cmap = plt.cm.tab10(np.arange(max(n_basin, 1)))
    for b in range(n_basin):
        m = part == b
        ax.scatter(scores[m, 0], scores[m, 1], s=2, alpha=0.2, color=cmap[b],
                   label=f"Basin {b+1}", rasterized=True)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("Latent plane by basin")
    lg = ax.legend(markerscale=6, fontsize=8)
    for h in lg.legend_handles:
        h.set_alpha(1)

    # row 1 col 1: diffusion map DC1-DC2 coloured by CryoSPARC class
    if ncol >= 2:
        ax = axes[1][1]
        dm = res["_dm"]
        ch = res["_cryo_hard"][dm["idx"]]
        ccmap = plt.cm.Set1(np.linspace(0, 1, max(len(class_names), 3)))
        for j, name in enumerate(class_names):
            m = ch == j
            ax.scatter(dm["dc"][m, 0], dm["dc"][m, 1], s=4, alpha=0.4,
                       color=ccmap[j], label=name, rasterized=True)
        ax.set_xlabel("diffusion coord 1"); ax.set_ylabel("diffusion coord 2")
        ax.set_title(f"Diffusion map (gap ~ {res['_kgap']} state(s))")
        lg = ax.legend(markerscale=3, fontsize=8)
        for h in lg.legend_handles:
            h.set_alpha(1)

    # row 1 col 2: diffusion eigenvalue spectrum with the gap
    if ncol >= 3:
        ax = axes[1][2]
        ev = res["diffusion_eigenvalues"]
        ax.plot(range(len(ev)), ev, "o-", color="darkgreen")
        ax.axvline(res["_kgap"] + 0.5, color="crimson", ls=":",
                   label=f"spectral gap ~ {res['_kgap']} state(s)")
        ax.set_xlabel("diffusion eigenvalue index")
        ax.set_ylabel("eigenvalue")
        ax.set_title("Diffusion spectrum")
        ax.legend(fontsize=8)
    for c in range(3, ncol):
        axes[1][c].axis("off")

    fig.suptitle(f"{label}: within-basin substates (Opt 3) + diffusion map "
                 f"(Opt 4)  |  {res['n_particles']:,} particles  |  "
                 f"{n_basin} basin(s)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(outdir, f"within_basin_{label}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


def write_summary(results, outdir):
    path = os.path.join(outdir, "within_basin_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Within-basin substates + hierarchical uncertainty\n\n")
        f.write("Two model-free Stage-2 tests (Option 3 local density peaks, "
                "Option 4 diffusion map) run on each cryoDRGN latent, plus "
                "Level-1 (basin population) and Level-2 (substate assignment) "
                "uncertainty.\n\n")
        for res in results:
            f.write(f"## {res['label']}  ({res['n_particles']:,} particles, "
                    f"zdim {res['zdim']})\n\n")
            f.write(f"- **Stage 1 basins:** {res['n_basins']} "
                    f"(populations "
                    + ", ".join(f"B{b+1}={p:.2f}±{s:.2f}"
                                for b, (p, s) in enumerate(zip(
                                    res['basin_populations'],
                                    res['basin_population_std']))) + ")\n")
            f.write(f"- **Diffusion-map spectral gap:** ~"
                    f"{res['diffusion_spectral_gap_k']} metastable state(s); "
                    f"|corr(DC1,PC1)| = {res['diffusion_dc1_pc1_corr']:.2f}\n")
            f.write("- **Within-basin substates (Option 3):**\n\n")
            f.write("  | basin | particles | sub-wells | deepest sub-barrier "
                    "(kT) | split-half reproducible |\n")
            f.write("  |---|---|---|---|---|\n")
            for b in res["substates"]:
                rep = b["reproducibility"]
                f.write(f"  | {b['basin']} | {b['n_particles']:,} | "
                        f"{b['n_sub_wells']} | "
                        f"{b['deepest_sub_barrier_kt']:.2f} | "
                        f"{rep['reproducible']} "
                        f"(cos={rep['axis_cos']}, corr={rep['curve_corr']}) |\n")
            f.write("\n")
        f.write("\n### How to read this\n")
        f.write("- A basin with **1 sub-well** = one structural blob = no "
                "substate to chase (Stage 2 stops).\n")
        f.write("- A basin with **>=2 reproducible sub-wells** (split-half "
                "agrees, high axis cos + curve corr) = a real candidate "
                "substate split -> run a **focused hetero-refine K = #sub-wells** "
                "on that basin's exported particles.\n")
        f.write("- Diffusion spectral gap is an *independent* count of "
                "metastable states; if it matches the basin count, the "
                "Stage-1 picture is robust.\n")
    print(f"[write] {path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", action="append", required=True,
                    help='"LABEL:z.pkl:passthrough.cs:cs:protein_idx(comma)"')
    ap.add_argument("--n-dummies", type=int, default=6)
    ap.add_argument("--barrier-kt", type=float, default=0.5,
                    help="Stage-1 basin merge threshold (kT)")
    ap.add_argument("--sub-barrier", type=float, default=0.5,
                    help="Stage-2 within-basin sub-well merge threshold (kT)")
    ap.add_argument("--min-pop", type=float, default=0.01,
                    help="drop Stage-1 basins below this particle fraction")
    ap.add_argument("--min-basin", type=int, default=2000,
                    help="skip within-basin analysis below this particle count")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--dm-sample", type=int, default=4000,
                    help="subsample size for the diffusion map (O(n^2) memory)")
    ap.add_argument("--eps-scale", type=float, default=1.0)
    ap.add_argument("--grid", type=int, default=140)
    ap.add_argument("--bw-scale", type=float, default=1.0)
    ap.add_argument("--sub", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--outdir", default="results_cryodrgn/within_basin_substates")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    results = []
    for spec in args.dataset:
        label, z_path, pass_cs, cs, idx = spec.split(":")
        protein_idx = [int(v) for v in idx.split(",")]
        res = analyse(label, z_path, pass_cs, cs, protein_idx,
                      args.n_dummies, args)
        plot_dataset(res, args.outdir)
        results.append(res)

    write_summary(results, args.outdir)
    clean = [{k: v for k, v in r.items() if not k.startswith("_")}
             for r in results]
    with open(os.path.join(args.outdir, "within_basin_metrics.json"),
              "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2)
    print(f"\n[done] outputs in {args.outdir}")


if __name__ == "__main__":
    main()
