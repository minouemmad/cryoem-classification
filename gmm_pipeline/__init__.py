"""GCER classification-uncertainty pipeline (GMM in posterior space)."""
from .data_io import load_posteriors
from .preprocess import alr_transform, simplex_drop_last
from .gmm_fit import fit_gmm, gmm_diagnostics
from .confusion import (
    monte_carlo_confusion,
    bhattacharyya_pairwise,
    hard_assignment_confusion,
    analytical_pairwise_confusion,
    analytical_multiclass_confusion,
)
from .uncertainty import (
    observed_populations,
    deconvolve_populations,
    bootstrap_population_ci,
    bootstrap_population_ci_analytical,
)
from .repetition import class_repetition_analysis
