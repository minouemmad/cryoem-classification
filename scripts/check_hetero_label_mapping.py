from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment

LABELS = ["P6", "P7", "P8"]
lab2int = {l: i for i, l in enumerate(LABELS)}

idx = pd.read_csv("results_J1069/exports_combined/combined_J1069_w1442_class_index.csv")
idx["lab"] = idx["class"].map(lab2int)
input_label = {int(u): int(l) for u, l in zip(idx["uid"], idx["lab"])}

branch_dir = Path("results_J1069/cryosparc_outputs/with_1442_weights")
hetero = branch_dir / "hetero_refinement"
hetero_files = [
    hetero / "P6" / "cryosparc_P25_J3571_class_00_00102_particles.cs",
    hetero / "P7" / "cryosparc_P25_J3571_class_01_00102_particles.cs",
    hetero / "P8" / "cryosparc_P25_J3571_class_02_00102_particles.cs",
]

new_label = {}
for cls_idx, f in enumerate(hetero_files):
    a = np.load(f)
    for u in a["uid"]:
        new_label[int(u)] = cls_idx

common = [u for u in new_label if u in input_label]
a = np.array([input_label[u] for u in common])
b = np.array([new_label[u] for u in common])

C = contingency_matrix(a, b)
print("Raw contingency (rows=orig P6/P7/P8, cols=hetero class_00/01/02):")
print(C)

r, c = C.shape
m = max(r, c)
Cs = np.zeros((m, m))
Cs[:r, :c] = C
row, col = linear_sum_assignment(-Cs)
mapping = {int(cand): int(ref) for ref, cand in zip(row, col)}
print("Hungarian mapping (hetero class id -> original label):")
for cand in sorted(mapping):
    print(f"  class_{cand:02d} -> {LABELS[mapping[cand]]}")

mapped = np.array([mapping[int(x)] for x in b])
Cm = contingency_matrix(a, mapped)
print("\nMapped contingency (rows=orig, cols=mapped new P6/P7/P8):")
print(Cm)
