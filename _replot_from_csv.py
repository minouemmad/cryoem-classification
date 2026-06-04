"""Re-plot summary figures from saved CSVs without re-running the full GMM."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gmm_pipeline.plots import (
    plot_class_table, plot_summary_table, plot_population_ci
)

p = argparse.ArgumentParser()
p.add_argument("outdir")
args = p.parse_args()
out = Path(args.outdir)

pop = pd.read_csv(out / "conformational_populations.csv")
labels = pop["class"].tolist()
pi_obs = pop["observed_csparc_hard"].values
pi_corr_a = pop["corrected_analytical"].values
pi_corr_lo = pop["corrected_lo"].values
pi_corr_hi = pop["corrected_hi"].values
pi_corr_std = pop["corrected_std_boot"].values
C_mc = pd.read_csv(out / "confusion_montecarlo.csv", index_col=0).values
C_an = pd.read_csv(out / "confusion_multiclass_analytical.csv", index_col=0).values
overlap = pd.read_csv(out / "class_overlap_bhattacharyya.csv", index_col=0).values
diag = json.load(open(out / "gmm_diagnostics.json"))
K = len(labels)
N = len(np.load(out / "responsibilities.npy"))
title = f"{out.name}  \u00b7  K={K}  \u00b7  N={N:,}  \u00b7  BIC={diag['bic']:.0f}"

plot_population_ci(pi_obs, pi_corr_a, pi_corr_std,
                   labels=labels, out=out / "conformational_populations.png")

plot_class_table(
    labels=labels, pi_obs=pi_obs, pi_corr=pi_corr_a,
    pi_corr_std=pi_corr_std, pi_corr_lo=pi_corr_lo, pi_corr_hi=pi_corr_hi,
    confusion_mc=C_mc, confusion_analytical=C_an,
    title=title, out=out / "summary_class_table.png",
)

plot_summary_table(
    labels=labels, pi_obs=pi_obs, pi_corr=pi_corr_a,
    pi_corr_mean=pi_corr_a, pi_corr_std=pi_corr_std,
    pi_corr_lo=pi_corr_lo, pi_corr_hi=pi_corr_hi,
    confusion_mc=C_mc, confusion_analytical=C_an,
    bhatt_overlap=overlap, diag=diag,
    title=title, out=out / "full_summary_table.png",
)
print(f"Plots regenerated in {out}/")
