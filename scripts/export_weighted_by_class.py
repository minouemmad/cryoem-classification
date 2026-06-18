"""Export per-conformation particle sets weighted by class posterior probability.

Motivation (Hunt, Jun 2026): instead of a hard ``max_resp > 0.9`` cut that keeps
only the most confident particles (and biases toward conformational extremes),
weight *every* protein particle's contribution to the per-class map by its
probability of belonging to that class. This is the M-step of a mixture-of-
volumes EM and uses all 230k particles:

    V_k = argmin_V  sum_n  w_nk || y_n - P(phi_n) V ||^2

The per-particle weight is injected into the CryoSPARC per-particle scale field
(``alignments3D/alpha`` by default), which Homogeneous Reconstruction Only /
Refinement applies directly during back-projection. Import the resulting .cs
(particle + passthrough) and run "Homogeneous Reconstruction Only".

Sharpening
----------
Raw posteriors in this CFTR data are near-uniform (mean max-post ~0.36), so
beta=1 weighting collapses every class map toward the consensus. A temperature
exponent beta sharpens the responsibilities and trades a little bias for real
class separation:

    w_nk = r_nk^beta / sum_j r_nj^beta

    beta = 1   -> John's exact unbiased soft weighting (consensus-collapsing here)
    beta -> inf -> recovers hard argmax assignment
    beta = 2..8 -> separation sweet spot for near-degenerate mixtures

Pass several values to ``--beta`` to sweep them in one run.

Cross-job posterior transplant
------------------------------
The particle stack (``--cs``) and the *source of the class probabilities*
(``--weights-cs``) need not be the same job. Pass ``--weights-cs`` to weight one
refinement's particles by another refinement's posteriors, matched by ``uid``.
This realises "weight the J1069 particles by the J1442 probabilities": J1069's
posteriors are near one-hot (overconfident), so its own weights behave like a
hard cut; J1442 is an honest soft posterior of the same particles. The exported
particle blob is the ``--cs`` stack (J1069), the poses/scale come from
``--passthrough-cs`` (J1442), and only the weights are taken from
``--weights-cs`` (J1442). Particles missing from any of the three are dropped.

Usage
-----
    # same-job soft weighting (original behaviour)
    python scripts/export_weighted_by_class.py \
        --cs data/cryosparc_P25_J1442_00000_particles.cs \
        --passthrough-cs data/cryosparc_P25_J1442_passthrough_particles_all_classes.cs \
        --n-dummies 6 --beta 1 2 4 --outdir results_J1442

    # cross-job: J1069 particles weighted by J1442 posteriors (J1442 poses)
    python scripts/export_weighted_by_class.py \
        --cs data/cryosparc_P25_J1069_00042_particles.cs \
        --weights-cs data/cryosparc_P25_J1442_00000_particles.cs \
        --passthrough-cs data/cryosparc_P25_J1442_passthrough_particles_all_classes.cs \
        --n-dummies 6 --beta 1 2 4 --outdir results_J1069
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gmm_pipeline import load_posteriors


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cs", required=True,
                   help="CryoSPARC *_particles.cs file = the particle STACK to "
                        "export (image refs). Also the posterior source unless "
                        "--weights-cs is given.")
    p.add_argument("--weights-cs", default=None,
                   help="Optional: take the class posteriors from THIS .cs "
                        "instead of --cs, matched to --cs by uid. Use to weight "
                        "one job's particles by another job's probabilities "
                        "(e.g. J1069 stack weighted by J1442 posteriors).")
    p.add_argument("--assign-cs", default=None,
                   help="Optional HARD-ASSIGNMENT source: restrict each class "
                        "export to particles whose argmax class in THIS .cs "
                        "equals that class (matched by uid). The scale factor "
                        "still comes from the posterior source. Without this, "
                        "every particle contributes to every class (full soft). "
                        "With it you get disjoint per-class subsets (hard "
                        "membership) each scaled by the posterior.")
    p.add_argument("--name-prefix", default="weighted",
                   help="Output filename prefix (e.g. 'hardJ1069') so different "
                        "experiments in the same --outdir do not overwrite.")
    p.add_argument("--passthrough-cs", required=True,
                   help="Matching passthrough .cs (carries single-class "
                        "alignments3D poses + the per-particle scale field)")
    p.add_argument("--n-dummies", type=int, default=6)
    p.add_argument("--protein-idx", type=int, nargs="*", default=None)
    p.add_argument("--beta", type=float, nargs="+", default=[1.0],
                   help="Sharpening exponent(s) for the posterior weights "
                        "(1 = unbiased soft; >1 = sharper/separating). Sweepable.")
    p.add_argument("--scale-field", default="alignments3D/alpha",
                   help="Passthrough field to overwrite with the weight "
                        "(default alignments3D/alpha; ctf/scale also plausible)")
    p.add_argument("--combine", choices=["replace", "multiply", "none"],
                   default="replace",
                   help="replace = set scale := weight (John's plan); "
                        "multiply = keep original scale * weight (amplitude-preserving); "
                        "none = leave scale untouched (CryoSPARC auto-weighting "
                        "reference; use with --assign-cs for plain hard subsets)")
    p.add_argument("--min-weight", type=float, default=0.0,
                   help="Drop particles whose class weight is below this, to "
                        "shrink the job (default 0 = keep all protein particles)")
    p.add_argument("--export-star", action="store_true",
                   help="Also convert each weighted set to .star via pyem/csparc2star")
    p.add_argument("--pyem-python", default=None,
                   help="Python executable with pyem installed (for --export-star)")
    p.add_argument("--outdir", default="results")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.outdir) / "exports_weighted"
    out.mkdir(parents=True, exist_ok=True)

    posterior_source = args.weights_cs or args.cs
    print(f"[1/4] Loading posteriors from {posterior_source}")
    if args.weights_cs:
        print(f"      (cross-job: weights from {args.weights_cs}, "
              f"particle stack from {args.cs})")
    post = load_posteriors(posterior_source, protein_idx=args.protein_idx,
                           n_dummies=args.n_dummies)
    prot = post.protein_only()
    labels = [f"P{int(c)}" for c in post.protein_idx]
    K = prot.n_protein
    r = prot.posterior                                  # (N, K) protein-only posteriors
    print(f"      N_protein={len(prot.uid):,}  K_protein={K}  labels={labels}")
    print(f"      mean posterior per class: {np.round(r.mean(axis=0), 4)}")
    print(f"      mean max-posterior (confidence): {r.max(axis=1).mean():.4f}")

    # Optional HARD membership: argmax class from a (possibly different) job,
    # aligned to prot.uid. -1 = particle absent from the assignment job.
    prot_assign = None
    if args.assign_cs:
        print(f"[1b/4] Loading HARD class assignments from {args.assign_cs}")
        a_post = load_posteriors(args.assign_cs, protein_idx=args.protein_idx,
                                 n_dummies=args.n_dummies).protein_only()
        assign_by_uid = {int(u): int(c)
                         for u, c in zip(a_post.uid, a_post.hard_class)}
        prot_assign = np.array([assign_by_uid.get(int(u), -1)
                                for u in prot.uid])
        amatched = int((prot_assign >= 0).sum())
        acounts = {labels[k]: int((prot_assign == k).sum()) for k in range(K)}
        print(f"       matched {amatched:,} of {len(prot.uid):,} particles; "
              f"hard class counts: {acounts}")

    print(f"[2/4] Loading passthrough {args.passthrough_cs}")
    pass_orig = np.load(args.passthrough_cs)
    if "uid" not in pass_orig.dtype.names:
        sys.exit("ERROR: passthrough .cs has no uid field; cannot align weights.")
    if args.scale_field not in pass_orig.dtype.names:
        sys.exit(f"ERROR: scale field '{args.scale_field}' not in passthrough. "
                 f"Available scale-like fields: "
                 f"{[n for n in pass_orig.dtype.names if 'alpha' in n or 'scale' in n]}")
    pass_uid_to_row = {int(u): i for i, u in enumerate(pass_orig["uid"])}

    cs_orig = np.load(args.cs)
    cs_uid_to_row = {int(u): i for i, u in enumerate(cs_orig["uid"])}

    scale_dtype = pass_orig.dtype[args.scale_field]

    manifest = []
    for beta in args.beta:
        # Sharpened, per-particle-normalised class weights.
        rb = np.power(np.clip(r, 1e-12, None), beta)
        w = rb / rb.sum(axis=1, keepdims=True)          # (N, K), rows sum to 1
        print(f"[3/4] beta={beta:g}: effective N per class (sum of weights): "
              f"{np.round(w.sum(axis=0)).astype(int)}")

        for k in range(K):
            wk = w[:, k]
            keep = wk > args.min_weight
            if prot_assign is not None:
                # Disjoint hard subset: only particles argmax-assigned to k.
                keep = keep & (prot_assign == k)
            # Keep UIDs as Python ints in lists: UIDs can exceed 2**63, and a
            # numpy array would silently promote them to float64 and lose the
            # low bits, breaking the uid->row lookups.
            uids_keep = [int(u) for u in prot.uid[keep]]
            w_by_uid = {u: float(val) for u, val in zip(uids_keep, wk[keep])}

            # Particles present in BOTH passthrough and particle blob, in order.
            matched = [u for u in uids_keep
                       if u in pass_uid_to_row and u in cs_uid_to_row]
            if not matched:
                print(f"        {labels[k]}: no UIDs matched passthrough — skipping")
                continue
            pass_rows = np.array([pass_uid_to_row[u] for u in matched])
            cs_rows = np.array([cs_uid_to_row[u] for u in matched])
            new_scale = np.array([w_by_uid[u] for u in matched])

            pass_subset = pass_orig[pass_rows].copy()
            if args.combine == "replace":
                pass_subset[args.scale_field] = new_scale.astype(scale_dtype.base)
            elif args.combine == "multiply":  # preserve original amplitude calibration
                orig = pass_subset[args.scale_field].astype(np.float64)
                pass_subset[args.scale_field] = (orig * new_scale).astype(scale_dtype.base)
            # combine == "none": leave the scale field untouched (CryoSPARC's
            # own per-particle scale/auto-weighting reference).

            tag = f"b{beta:g}_{labels[k]}"
            pass_name = out / f"{args.name_prefix}_{tag}_passthrough.cs"
            with open(pass_name, "wb") as fh:
                np.save(fh, pass_subset)

            # Matching particle blob (unmodified poses/refs), needed for import
            # and for csparc2star (which consumes particles + passthrough).
            cs_name = out / f"{args.name_prefix}_{tag}.cs"
            with open(cs_name, "wb") as fh:
                np.save(fh, cs_orig[cs_rows])

            eff_n = float(new_scale.sum())
            print(f"        {labels[k]}: {len(pass_rows):,} particles  "
                  f"effective N={eff_n:,.0f}  -> {cs_name.name}")
            manifest.append({
                "beta": beta, "class": labels[k],
                "n_particles": len(pass_rows),
                "effective_n": eff_n,
                "mean_weight": float(new_scale.mean()),
                "cs": cs_name.name, "passthrough": pass_name.name,
            })

            if args.export_star:
                from run_pipeline import _convert_cs_to_star
                star_name = out / f"{args.name_prefix}_{tag}.star"
                try:
                    cmd = _convert_cs_to_star(cs_name, pass_name, star_name,
                                              pyem_python=args.pyem_python)
                    print(f"          star -> {star_name.name} (via {cmd})")
                except RuntimeError as exc:
                    print(f"          star conversion failed:\n{exc}")

    print("[4/4] Writing manifest")
    manifest_name = out / f"{args.name_prefix}_export_manifest.csv"
    pd.DataFrame.from_records(manifest).to_csv(manifest_name, index=False)
    print(f"      manifest -> {manifest_name}")
    print(f"\nDone. Per-class weighted sets in {out.resolve()}")
    print("Import each weighted_*.cs (+ _passthrough.cs) and run "
          "'Homogeneous Reconstruction Only'.")


if __name__ == "__main__":
    main()
