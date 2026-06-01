"""Load CryoSPARC .cs particle files and extract the class-posterior matrix."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


POSTERIOR_FIELD = "alignments3D_multi/class_posterior"
HARD_CLASS_FIELD = "alignments3D_multi/class"
UID_FIELD = "uid"


@dataclass
class Posteriors:
    uid: np.ndarray            # (N,)
    posterior: np.ndarray      # (N, K) rows ~sum to 1
    hard_class: np.ndarray     # (N,) integer in [0, K)
    protein_idx: np.ndarray    # indices into K that are protein classes
    dummy_idx: np.ndarray      # indices into K that are dummy classes

    @property
    def n_classes(self) -> int:
        return self.posterior.shape[1]

    @property
    def n_protein(self) -> int:
        return len(self.protein_idx)

    def protein_only(self) -> "Posteriors":
        """Keep particles whose hard assignment is a protein class; renormalize
        their posteriors over protein classes only."""
        mask = np.isin(self.hard_class, self.protein_idx)
        sub_post = self.posterior[mask][:, self.protein_idx]
        sub_post = sub_post / sub_post.sum(axis=1, keepdims=True).clip(min=1e-12)
        # remap hard class to position within protein_idx
        remap = {int(c): i for i, c in enumerate(self.protein_idx)}
        new_hard = np.array([remap[int(c)] for c in self.hard_class[mask]])
        return Posteriors(
            uid=self.uid[mask],
            posterior=sub_post,
            hard_class=new_hard,
            protein_idx=np.arange(len(self.protein_idx)),
            dummy_idx=np.array([], dtype=int),
        )


def load_posteriors(
    cs_path: str,
    protein_idx: Sequence[int] | None = None,
    n_dummies: int | None = 6,
) -> Posteriors:
    """Load a CryoSPARC .cs particle file.

    Parameters
    ----------
    cs_path : path to ``cryosparc_*_particles.cs``
    protein_idx : explicit zero-based indices of protein classes; if None,
        the last ``K - n_dummies`` classes are treated as protein.
    n_dummies : number of leading dummy classes (ignored if ``protein_idx`` given).
    """
    cs = np.load(cs_path)
    post = np.asarray(cs[POSTERIOR_FIELD], dtype=np.float64)
    # numerical normalization (rows may be off by ~1e-5)
    post = post / post.sum(axis=1, keepdims=True).clip(min=1e-12)

    # CryoSPARC's ``alignments3D_multi/class`` is per-class metadata (shape
    # (N, K)), not a hard label, so always derive hard assignment from posterior.
    hard = post.argmax(axis=1)

    uid = np.asarray(cs[UID_FIELD]) if UID_FIELD in cs.dtype.names else np.arange(len(post))

    K = post.shape[1]
    if protein_idx is None:
        nd = 0 if n_dummies is None else int(n_dummies)
        protein_idx = np.arange(nd, K)
    protein_idx = np.array(sorted(int(i) for i in protein_idx))
    dummy_idx = np.array([i for i in range(K) if i not in protein_idx])

    return Posteriors(uid=uid, posterior=post, hard_class=hard,
                      protein_idx=protein_idx, dummy_idx=dummy_idx)
