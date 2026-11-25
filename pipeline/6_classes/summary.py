"""Model / knob comparison: every finished run of a dataset against its base run, at class count K.
Per run: coordinates kept, elbow, off-blend rule, class sizes, ARI of the particle labels vs the base run, Hungarian-matched class-map cc
vs the base run (cell half maps, n-weighted, minus the mean), and the share of the main axis found in rounds 1-6 of the search.
Usage: summary.py DS [BASE=rae_base] [K=3]"""
import glob, json, os, sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score as ari
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
from maps import cc, class_maps

ds = sys.argv[1]; base = sys.argv[2] if len(sys.argv) > 2 else "rae_base"; K = int(sys.argv[3]) if len(sys.argv) > 3 else 3; P = os.environ["WORK"] + f"/pipe/{ds}/"


def load(run):
    z, c = np.load(P + f"{run}/mc/MP_model_labels.npz"), np.load(P + f"{run}/mc/MP_model_cells.npz"); n = np.bincount(c["cell"], minlength=len(z[f"K{K}"])).astype(float)
    X = (c["A"] + c["B"]) / 2; M = class_maps(X, n, z[f"K{K}"], K)
    return z[f"K{K}"][z["cell"]], M - (X * n[:, None]).sum(0) / n.sum()


Lb, Mb = load(base)
for f in sorted(glob.glob(P + "*/mc/MP_model_labels.npz")):
    run = f.split("/")[-3]; j = json.load(open(os.path.dirname(f) + "/MP.json"))["model"]; L, M = load(run)
    C = np.array([[cc(a, b) for b in M] for a in Mb]); r, c = linear_sum_assignment(-C)
    it = json.load(open(P + run + "/iter2/iter.json")) if os.path.exists(P + run + "/iter2/iter.json") else []; pct = " ".join("%.1f" % x["pct_of_main"] for x in it[:6])
    print(f"{run:28s} coords {j['n_coords']:2d}  elbow {j['elbow']}  offblend {j['offblend_halving']}  K{K} sizes {sorted(np.bincount(L, minlength=K).tolist(), reverse=True)}"
          f"  ARI vs {base} {ari(Lb, L):.2f}  map cc {' '.join(f'{C[i, k]:.2f}' for i, k in zip(r, c))}  | % of main r1-6: {pct}")
