"""Single source of truth for CFTR conformational-class display names.

Every cryoDRGN figure script imports this so class labels are consistent and
publishable. Maps the CryoSPARC protein-class index (P6, P7, ...) to a
biological name for each dataset.

    import class_names as cnames
    cnames.label("J1497", 8)      -> "P8 (VshapedMix)"
    cnames.labels_for("J1442", [6, 7, 8])
    cnames.guess_dataset(path)    -> "J1442" | "J1497" | "J264" | "J325" | None
"""
from __future__ import annotations

# J1442 (3-class, P6-P8) and J1497 (5-class, P6-P10) are the SAME particles;
# names indexed by CryoSPARC protein-class index.
J1497_NAMES = {
    6: "NBD1LessMix-Ablated",
    7: "NBD1LessWide-Ablated",
    8: "VshapedMix",
    9: "NBD2Less-Ablated",
    10: "AltNBD1-ArdeconComposite-Ablated",
}
J1442_NAMES = {6: J1497_NAMES[6], 7: J1497_NAMES[7], 8: J1497_NAMES[8]}

# J264 (9-class, P6-P14) and J325/J326 (6-class, P6-P11): CFTR conformations.
J264_NAMES = {
    6: "SC", 7: "AC", 8: "AO", 9: "SEPD", 10: "AEPD",
    11: "V-shaped", 12: "NBD1-less", 13: "NBD2-less", 14: "NBD1-less-wide",
}
J325_NAMES = {6: "SC", 7: "AC", 8: "AO", 9: "SEPD", 10: "AEPD", 11: "V-shaped"}

DATASETS = {
    "J1442": J1442_NAMES,
    "J1497": J1497_NAMES,
    "J264": J264_NAMES,
    "J325": J325_NAMES,
    "J326": J325_NAMES,  # J326 = debiased rerun of the 6-class refine
}


def name_for(dataset: str | None, idx: int) -> str | None:
    """Biological name for a class index, or None if unknown."""
    return DATASETS.get(dataset or "", {}).get(idx)


def label(dataset: str | None, idx: int) -> str:
    """'P8 (VshapedMix)' style label; falls back to 'P8' if name unknown."""
    nm = name_for(dataset, idx)
    return f"P{idx} ({nm})" if nm else f"P{idx}"


def labels_for(dataset: str | None, protein_idx) -> list[str]:
    return [label(dataset, int(i)) for i in protein_idx]


def guess_dataset(*paths) -> str | None:
    """Infer the dataset key from any path/string. J1497 before J1442 so the
    5-class job is not shadowed by a shared substring."""
    s = " ".join(str(p) for p in paths if p is not None)
    for key in ("J1497", "J1442", "J326", "J325", "J264"):
        if key in s:
            return key
    return None
