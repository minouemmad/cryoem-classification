"""Re-plot summary figures from saved CSVs without re-running the full GMM."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gmm_pipeline.plots import (
    plot_class_table, plot_population_ci, plot_population_comparison,
)

p = argparse.ArgumentParser()
p.add_argument("outdir")
args = p.parse_args()
out = Path(args.outdir)
conf_dir, pop_dir, gmm_dir = out / "confusion", out / "populations", out / "gmm"

pop = pd.read_csv(pop_dir / "conformational_populations.csv")
labels = pop["class"].tolist()
pi_obs = pop["observed_csparc_hard"].values
pi_corr_soft = pop["corrected_soft_posterior"].values
pi_corr_lo = pop["corrected_lo"].values
pi_corr_hi = pop["corrected_hi"].values
pi_corr_std = pop["corrected_std_boot"].values

# Per-matrix corrections + accuracies (written by run_pipeline).
allm = pd.read_csv(pop_dir / "population_corrections_all_matrices.csv")
keys = [c[len("corrected_"):] for c in allm.columns if c.startswith("corrected_")]
corrections = {k: allm[f"corrected_{k}"].values for k in keys}
accuracies = {k: allm[f"accuracy_{k}"].values for k in keys}

overlap = pd.read_csv(conf_dir / "class_overlap_bhattacharyya.csv", index_col=0).values
diag = json.load(open(gmm_dir / "gmm_diagnostics.json"))
K = len(labels)
N = len(np.load(gmm_dir / "responsibilities.npy"))
title = f"{out.name}  \u00b7  K={K}  \u00b7  N={N:,}  \u00b7  BIC={diag['bic']:.0f}"

plot_population_ci(pi_obs, pi_corr_soft, pi_corr_std,
                   labels=labels, out=pop_dir / "conformational_populations.png")
plot_population_comparison(pi_obs, corrections, labels=labels,
                           out=pop_dir / "population_corrections_all_matrices.png")
plot_class_table(
    labels=labels, pi_obs=pi_obs,
    corrections=corrections, accuracies=accuracies, primary="soft",
    pi_corr_std=pi_corr_std, pi_corr_lo=pi_corr_lo, pi_corr_hi=pi_corr_hi,
    extra_metrics={"Max\noverlap": overlap.max(axis=1)},
    title=title, out=pop_dir / "summary_class_table.png",
)
print(f"Plots regenerated in {pop_dir}/")
