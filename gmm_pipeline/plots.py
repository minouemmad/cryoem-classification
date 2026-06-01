"""Lightweight diagnostic plots for the GMM uncertainty pipeline."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion(C: np.ndarray, labels=None, title="Confusion matrix", out=None):
    fig, ax = plt.subplots(figsize=(4 + 0.4 * len(C), 3.5 + 0.4 * len(C)))
    im = ax.imshow(C, vmin=0, vmax=1, cmap="viridis")
    K = len(C)
    labels = labels or [f"C{i}" for i in range(K)]
    ax.set_xticks(range(K), labels)
    ax.set_yticks(range(K), labels)
    ax.set_xlabel("Assigned")
    ax.set_ylabel("True")
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center",
                    color="white" if C[i,j] < 0.5 else "black", fontsize=9)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return fig


def plot_population_ci(observed, corrected, lo, hi, labels=None, out=None):
    K = len(observed)
    x = np.arange(K)
    labels = labels or [f"C{i}" for i in range(K)]
    fig, ax = plt.subplots(figsize=(1.4 * K + 2, 4))
    ax.bar(x - 0.18, observed, width=0.35, label="Observed", color="#888")
    ax.bar(x + 0.18, corrected, width=0.35, label="Corrected", color="#3a7")
    ax.errorbar(x + 0.18, corrected,
                yerr=[corrected - lo, hi - corrected],
                fmt="none", ecolor="black", capsize=4)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Population fraction")
    ax.set_title("Conformational populations  (corrected = (Cᵀ)⁻¹ pi_obs)")
    ax.legend()
    fig.tight_layout()
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return fig


def plot_repetition(r_values, mapped_weights, labels=None, out=None):
    K = mapped_weights.shape[1]
    labels = labels or [f"C{i}" for i in range(K)]
    fig, ax = plt.subplots(figsize=(6, 4))
    for k in range(K):
        ax.plot(r_values, mapped_weights[:, k], "o-", label=labels[k])
    ax.set_xlabel("# extra (duplicate) components r")
    ax.set_ylabel("Aggregated weight of base class")
    ax.set_title("Class-repetition analysis")
    ax.legend()
    fig.tight_layout()
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return fig
