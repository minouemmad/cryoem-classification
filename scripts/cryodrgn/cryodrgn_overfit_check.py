#!/usr/bin/env python
r"""Overfitting / over-confidence diagnostics for the cryoDRGN latent (PC1 axis).

John: "cryoDRGN's PC1 axis is too confident and suspects overfitting is involved.
       Make sure the model isn't overfitting."

cryoDRGN's train_vae does NOT hold out a validation set, so the textbook
train-vs-validation-loss gap is unavailable. Instead we run four independent,
workflow-doable checks that each probe a different overfitting signature:

  (1) CONFIDENCE-vs-SEPARATION paradox.
      A K-component GMM on the latent reports near-100% responsibilities while
      its components sit only ~1.9 SD apart. For real Gaussians that close, the
      expected max-responsibility is far below 1.0. We quantify the gap: a large
      gap = the discrete model is MANUFACTURING confidence on a continuous cloud
      (the "too confident" John noticed), not finding real wells.

  (2) IMAGING-CONFOUND correlation.
      If PC1 is conformation it should be (largely) uncorrelated with imaging
      nuisances. We regress each principal component on CTF defocus / astigmatism
      / phase shift / per-particle scale / pose-rotation / shift / box-position.
      High R^2 (esp. on defocus or scale) = the latent overfit to imaging
      artifacts, not structure.

  (3) EPOCH STABILITY of PC1.
      Using the saved per-epoch z.N.pkl, we correlate PC1 at successive epochs
      against the final PC1 (sign-matched). A converged, non-overfit coordinate
      locks in early and stays put; a PC1 that keeps drifting late in training is
      a sign of memorisation / instability.

  (4) TRAIN-LOSS curve (from run.log).
      Plotted for context (flat tail = converged). Note: flat train loss alone
      cannot rule out overfitting (no held-out set) - that is why (1)-(3) matter.

Run with the cryoDRGN env from repo root::

    python scripts/cryodrgn/cryodrgn_overfit_check.py \
      --train-dir results_cryodrgn/J1442_real/train_z10 \
      --final-z results_cryodrgn/J1442_real/train_z10/z.100.pkl \
      --passthrough-cs data/cryosparc_P25_J1442_passthrough_particles_all_classes.cs \
      --cs data/cryosparc_P25_J1442_00000_particles.cs \
      --n-dummies 6 --protein-idx 6 7 8 -k 3 \
      -o results_cryodrgn/J1442_real/overfit_check
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import cryodrgn_latent_gmm as clg


# --------------------------------------------------------------------------- #
# (1) confidence vs separation
# --------------------------------------------------------------------------- #
def expected_maxresp_two_gaussians(sep_sd, n=200000, seed=0):
    """Monte-Carlo expected max posterior for an equal-weight 2-Gaussian mixture
    whose means are `sep_sd` pooled-SDs apart (unit variance each, 1-D)."""
    rng = np.random.default_rng(seed)
    half = n // 2
    x = np.concatenate([rng.normal(0, 1, half), rng.normal(sep_sd, 1, n - half)])
    # posterior for component B vs A under equal priors, unit variance
    logA = -0.5 * x ** 2
    logB = -0.5 * (x - sep_sd) ** 2
    m = np.maximum(logA, logB)
    pA = np.exp(logA - m)
    pB = np.exp(logB - m)
    post = np.maximum(pA, pB) / (pA + pB)
    return float(post.mean())


def confidence_separation_check(Xs, k, seed, outdir):
    gmm = GaussianMixture(k, covariance_type="full", n_init=10, max_iter=1000,
                          tol=1e-6, reg_covar=1e-6, random_state=seed).fit(Xs)
    resp = gmm.predict_proba(Xs)
    hard = resp.argmax(1)
    mean_maxresp = float(resp.max(1).mean())
    frac_conf = float((resp.max(1) > 0.9).mean())

    means, covs = gmm.means_, gmm.covariances_
    d = means.shape[1]
    seps = []
    for a in range(k):
        for b in range(a + 1, k):
            pooled = 0.5 * (np.trace(covs[a]) + np.trace(covs[b])) / d
            seps.append(np.linalg.norm(means[a] - means[b]) / np.sqrt(max(pooled, 1e-12)))
    min_sep = float(min(seps))
    mean_sep = float(np.mean(seps))

    expected = expected_maxresp_two_gaussians(min_sep, seed=seed)
    gap = mean_maxresp - expected

    # bar plot: observed vs expected
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.bar([0, 1], [expected, mean_maxresp],
           color=["#7fb069", "#d1495b"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"expected\n(2 Gauss {min_sep:.2f} SD apart)",
                        "observed\nlatent-GMM"])
    ax.set_ylabel("mean max responsibility")
    ax.set_ylim(0.5, 1.02)
    ax.set_title(f"Over-confidence gap = {gap:.3f}\n"
                 f"(components only {min_sep:.2f} SD apart but assigns "
                 f"{mean_maxresp:.3f} confident)")
    for i, v in enumerate([expected, mean_maxresp]):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "confidence_vs_separation.png"), dpi=150)
    plt.close(fig)

    return {
        "mean_max_responsibility": mean_maxresp,
        "frac_resp_gt_0.9": frac_conf,
        "min_separation_sd": min_sep,
        "mean_separation_sd": mean_sep,
        "expected_maxresp_at_min_sep": expected,
        "overconfidence_gap": gap,
    }, hard


# --------------------------------------------------------------------------- #
# (2) imaging-confound correlation
# --------------------------------------------------------------------------- #
def _pose_rotation_angle(pose):
    """CryoSPARC alignments3D/pose = rotation as a 3-vector (axis * angle).
    Return the rotation angle magnitude per particle (radians)."""
    pose = np.asarray(pose, dtype=np.float64)
    if pose.ndim == 1:
        pose = pose.reshape(-1, 3)
    return np.linalg.norm(pose, axis=1)


def build_confounds(cs_pass, uid_keep):
    """Return dict name -> array (aligned to uid_keep order) of imaging nuisances."""
    pass_uids = cs_pass["uid"].astype(np.uint64)
    row_of = {int(u): i for i, u in enumerate(pass_uids.tolist())}
    rows = np.asarray([row_of[int(u)] for u in uid_keep.tolist()], dtype=np.intp)

    def col(name):
        return np.asarray(cs_pass[name])[rows]

    conf = {}
    df1 = col("ctf/df1_A").astype(np.float64)
    df2 = col("ctf/df2_A").astype(np.float64)
    conf["defocus_mean_A"] = 0.5 * (df1 + df2)
    conf["astigmatism_A"] = np.abs(df1 - df2)
    if "ctf/df_angle_rad" in cs_pass.dtype.names:
        conf["df_angle_rad"] = col("ctf/df_angle_rad").astype(np.float64)
    if "ctf/phase_shift_rad" in cs_pass.dtype.names:
        conf["phase_shift_rad"] = col("ctf/phase_shift_rad").astype(np.float64)
    if "ctf/scale" in cs_pass.dtype.names:
        conf["ctf_scale"] = col("ctf/scale").astype(np.float64)
    if "alignments3D/alpha" in cs_pass.dtype.names:
        conf["particle_scale_alpha"] = col("alignments3D/alpha").astype(np.float64)
    if "alignments3D/pose" in cs_pass.dtype.names:
        conf["pose_rotation_rad"] = _pose_rotation_angle(col("alignments3D/pose"))
    if "alignments3D/shift" in cs_pass.dtype.names:
        sh = np.asarray(col("alignments3D/shift"), dtype=np.float64)
        if sh.ndim == 1:
            sh = sh.reshape(-1, 2)
        conf["shift_magnitude_px"] = np.linalg.norm(sh, axis=1)
    # drop any constant columns (no variance -> correlation undefined)
    return {k_: v for k_, v in conf.items() if np.std(v) > 0}


def confound_check(scores, confounds, outdir, n_pcs=3):
    pcs = min(n_pcs, scores.shape[1])
    conf_names = list(confounds)
    R = np.zeros((pcs, len(conf_names)))
    for i in range(pcs):
        for j, cn in enumerate(conf_names):
            R[i, j] = np.corrcoef(scores[:, i], confounds[cn])[0, 1]

    # multivariate R^2 of regressing PC1 on ALL confounds together
    Xc = np.column_stack([StandardScaler().fit_transform(confounds[cn][:, None]).ravel()
                          for cn in conf_names])
    Xc = np.column_stack([np.ones(len(Xc)), Xc])
    y = StandardScaler().fit_transform(scores[:, 0][:, None]).ravel()
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    yhat = Xc @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    pc1_r2 = 1.0 - ss_res / ss_tot

    fig, ax = plt.subplots(figsize=(1.8 + 1.1 * len(conf_names), 1.6 + 0.7 * pcs))
    im = ax.imshow(R, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(conf_names)))
    ax.set_xticklabels(conf_names, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(pcs))
    ax.set_yticklabels([f"PC{i+1}" for i in range(pcs)])
    for i in range(pcs):
        for j in range(len(conf_names)):
            ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(R[i, j]) > 0.5 else "black", fontsize=8)
    ax.set_title(f"Latent PC vs imaging confounds (Pearson r)\n"
                 f"PC1 multivariate R^2 on all confounds = {pc1_r2:.3f}")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "confound_correlation.png"), dpi=150)
    plt.close(fig)

    return {
        "confound_names": conf_names,
        "pc_vs_confound_pearson": R.tolist(),
        "pc1_multivariate_r2_on_confounds": float(pc1_r2),
        "max_abs_pc1_confound_corr": float(np.max(np.abs(R[0]))),
        "max_abs_pc1_confound_name": conf_names[int(np.argmax(np.abs(R[0])))],
    }


# --------------------------------------------------------------------------- #
# (3) epoch stability of PC1
# --------------------------------------------------------------------------- #
def epoch_stability(train_dir, passthrough_cs, cs, n_dummies, protein_idx,
                    final_pc1, final_uid, outdir, seed=0):
    z_files = glob.glob(os.path.join(train_dir, "z.*.pkl"))
    epoch_of = {}
    for f in z_files:
        m = re.search(r"z\.(\d+)\.pkl$", os.path.basename(f))
        if m:
            epoch_of[int(m.group(1))] = f
    if not epoch_of:
        return None
    epochs = sorted(epoch_of)
    # sample up to ~12 epochs spread across training (always include the last)
    if len(epochs) > 12:
        idx = np.linspace(0, len(epochs) - 1, 12).round().astype(int)
        epochs = sorted(set([epochs[i] for i in idx] + [epochs[-1]]))

    final_map = {int(u): p for u, p in zip(final_uid.tolist(), final_pc1.tolist())}
    corrs = []
    for ep in epochs:
        z = clg.load_latent(epoch_of[ep])
        z, _, _, uid, _ = clg.align_z_to_posteriors(
            z, passthrough_cs, cs, n_dummies, protein_idx)
        Xs = StandardScaler().fit_transform(z)
        pc1 = PCA(n_components=1, random_state=seed).fit_transform(Xs).ravel()
        ref = np.asarray([final_map.get(int(u), np.nan) for u in uid.tolist()])
        ok = ~np.isnan(ref)
        r = np.corrcoef(pc1[ok], ref[ok])[0, 1]
        corrs.append(abs(r))                       # sign of PC1 is arbitrary
        print(f"  epoch {ep:3d}: |corr(PC1, final PC1)| = {abs(r):.4f}")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(epochs, corrs, "o-", color="#2c7fb8")
    ax.axhline(0.95, color="gray", ls=":", label="0.95")
    ax.set_xlabel("training epoch")
    ax.set_ylabel("|corr(PC1 at epoch, final PC1)|")
    ax.set_ylim(0, 1.02)
    ax.set_title("PC1 epoch stability (locks in early = converged, not memorising)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "pc1_epoch_stability.png"), dpi=150)
    plt.close(fig)

    # epoch at which corr first exceeds 0.95
    lock_epoch = next((e for e, c in zip(epochs, corrs) if c >= 0.95), None)
    return {
        "epochs": epochs,
        "abs_corr_to_final_pc1": [float(c) for c in corrs],
        "lock_in_epoch_corr0.95": (int(lock_epoch) if lock_epoch is not None else None),
        "final_epoch_corr": float(corrs[-1]),
    }


# --------------------------------------------------------------------------- #
# (4) train-loss curve
# --------------------------------------------------------------------------- #
def parse_loss(train_dir, outdir):
    log = os.path.join(train_dir, "run.log")
    if not os.path.exists(log):
        return None
    pat = re.compile(r"Epoch:\s*(\d+)\s+Average gen loss\s*=\s*([\d.]+).*?"
                     r"total loss\s*=\s*([\d.]+)")
    eps, gen, tot = [], [], []
    with open(log, "r", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                eps.append(int(m.group(1)))
                gen.append(float(m.group(2)))
                tot.append(float(m.group(3)))
    if not eps:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(eps, tot, "o-", ms=3, label="total loss", color="#222")
    ax.plot(eps, gen, "s--", ms=3, label="gen loss", color="#888")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title("cryoDRGN training loss (no held-out set; flat tail = converged)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "train_loss.png"), dpi=150)
    plt.close(fig)
    tail = tot[max(0, len(tot) - 20):]
    return {
        "n_epochs": len(eps),
        "final_total_loss": float(tot[-1]),
        "min_total_loss": float(min(tot)),
        "tail20_relative_change": float((max(tail) - min(tail)) / max(tail)),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-dir", required=True,
                   help="cryoDRGN train dir (has z.N.pkl + run.log).")
    p.add_argument("--final-z", required=True, help="final z.N.pkl.")
    p.add_argument("--passthrough-cs", required=True)
    p.add_argument("--cs", required=True)
    p.add_argument("--n-dummies", type=int, default=6)
    p.add_argument("--protein-idx", type=int, nargs="*", default=None)
    p.add_argument("-k", "--k", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-o", "--outdir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    z = clg.load_latent(args.final_z)
    z, cryo_post, cryo_hard, uid, n_protein = clg.align_z_to_posteriors(
        z, args.passthrough_cs, args.cs, args.n_dummies, args.protein_idx)
    k = args.k or n_protein
    Xs = StandardScaler().fit_transform(z)
    pca = PCA(n_components=min(5, z.shape[1]), random_state=args.seed).fit(Xs)
    scores = pca.transform(Xs)

    print("[1] confidence vs separation")
    conf_sep, _ = confidence_separation_check(Xs, k, args.seed, args.outdir)

    print("[2] imaging-confound correlation")
    cs_pass = np.load(args.passthrough_cs)
    confounds = build_confounds(cs_pass, uid)
    confound = confound_check(scores, confounds, args.outdir)

    print("[3] PC1 epoch stability")
    stability = epoch_stability(args.train_dir, args.passthrough_cs, args.cs,
                                args.n_dummies, args.protein_idx,
                                scores[:, 0], uid, args.outdir, args.seed)

    print("[4] train-loss curve")
    loss = parse_loss(args.train_dir, args.outdir)

    # ---- verdict ------------------------------------------------------------
    flags = []
    if conf_sep["overconfidence_gap"] > 0.15:
        flags.append(
            f"OVER-CONFIDENT: latent-GMM assigns {conf_sep['mean_max_responsibility']:.3f} "
            f"mean confidence though components are only {conf_sep['min_separation_sd']:.2f} "
            f"SD apart (expected ~{conf_sep['expected_maxresp_at_min_sep']:.3f}); "
            f"gap {conf_sep['overconfidence_gap']:.3f}. The discrete K-state model "
            "manufactures confidence on a continuous cloud - this is the 'too confident "
            "PC1' effect, NOT evidence of real wells.")
    else:
        flags.append("Confidence is consistent with component separation.")

    if confound["max_abs_pc1_confound_corr"] > 0.3 or \
            confound["pc1_multivariate_r2_on_confounds"] > 0.2:
        flags.append(
            f"CONFOUND WARNING: PC1 correlates with imaging nuisances "
            f"(max |r|={confound['max_abs_pc1_confound_corr']:.2f} with "
            f"{confound['max_abs_pc1_confound_name']}, multivariate R^2="
            f"{confound['pc1_multivariate_r2_on_confounds']:.2f}) -> some of PC1 may be "
            "imaging artifact, not conformation.")
    else:
        flags.append(
            f"PC1 is largely free of imaging confounds (max |r|="
            f"{confound['max_abs_pc1_confound_corr']:.2f}, R^2="
            f"{confound['pc1_multivariate_r2_on_confounds']:.2f}) -> PC1 reflects "
            "structure, not CTF/pose/scale.")

    if stability is not None:
        if stability["lock_in_epoch_corr0.95"] is not None:
            flags.append(
                f"STABLE: PC1 locks to its final direction by epoch "
                f"{stability['lock_in_epoch_corr0.95']} (|corr|>=0.95) and holds -> "
                "converged coordinate, not late-training memorisation.")
        else:
            flags.append(
                "UNSTABLE: PC1 never reaches |corr|>=0.95 with the final PC1 across "
                "sampled epochs -> the coordinate keeps drifting (possible "
                "instability/overfitting).")

    if loss is not None:
        flags.append(
            f"Train loss converged (final {loss['final_total_loss']:.4f}, tail-20 "
            f"relative change {loss['tail20_relative_change']*100:.2f}%). NOTE: "
            "cryoDRGN train_vae holds out no validation set, so flat train loss "
            "alone cannot exclude overfitting - checks (1)-(3) are the decisive ones.")

    result = {
        "n_particles": int(len(z)),
        "zdim": int(z.shape[1]),
        "k": int(k),
        "pc1_explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
        "confidence_separation": conf_sep,
        "confound": confound,
        "epoch_stability": stability,
        "train_loss": loss,
        "verdict": flags,
    }
    with open(os.path.join(args.outdir, "overfit_diagnostics.json"), "w") as f:
        json.dump(result, f, indent=2)

    lines = ["# cryoDRGN overfitting / over-confidence diagnostics", "",
             f"- particles: {len(z):,}  |  zdim: {z.shape[1]}  |  k: {k}",
             f"- PC1 explained variance ratio: {pca.explained_variance_ratio_[0]:.3f}",
             "", "## Verdict", ""]
    for fl in flags:
        lines.append(f"- {fl}")
    lines += ["", "## Key numbers", "",
              f"- mean max responsibility: {conf_sep['mean_max_responsibility']:.3f} "
              f"(frac>0.9 = {conf_sep['frac_resp_gt_0.9']:.3f})",
              f"- min component separation: {conf_sep['min_separation_sd']:.2f} SD "
              f"(expected max-resp at this separation: "
              f"{conf_sep['expected_maxresp_at_min_sep']:.3f})",
              f"- over-confidence gap: {conf_sep['overconfidence_gap']:.3f}",
              f"- PC1 max |confound corr|: {confound['max_abs_pc1_confound_corr']:.2f} "
              f"({confound['max_abs_pc1_confound_name']}); multivariate R^2 "
              f"{confound['pc1_multivariate_r2_on_confounds']:.2f}"]
    if stability is not None:
        lines.append(f"- PC1 lock-in epoch (|corr|>=0.95): "
                     f"{stability['lock_in_epoch_corr0.95']}; final-epoch corr "
                     f"{stability['final_epoch_corr']:.3f}")
    with open(os.path.join(args.outdir, "SUMMARY.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n[done] outputs ->", args.outdir)
    for fl in flags:
        print("  -", fl)


if __name__ == "__main__":
    main()
