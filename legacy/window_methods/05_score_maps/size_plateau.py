"""Label-free size criterion: how much does the map change as the Gaussian window grows? The 12 Gaussian soft-window maps (kern + kern2, one map each, centred on the rule's pick) at 10 A, same normalisat
Usage: size_plateau.py -> size_plateau_1117.log"""
import sys
from pathlib import Path
import numpy as np, mrcfile
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from mapdiff import envelope, lowpass
H = Path("/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn/J4001_gP25_WT/method_rank"); F = H / "fits"
apix = float(mrcfile.open(F / "fit_J2307.mrc").voxel_size.x)
rd = lambda n: mrcfile.read(F / (f"in_{n}.mrc" if n == "J1049" else f"fit_{n}.mrc")).astype(np.float32)
G = [("8.7k", "J2522"), ("11.6k", "J2525"), ("17.4k", "J2528"), ("23.1k", "J2531"), ("28.9k", "J2480"), ("34.5k", "J2534"), ("43.2k", "J2483"), ("57.7k", "J2486"),
     ("75.0k", "J2489"), ("92.0k", "J2537"), ("113k", "J2540"), ("155k", "J2543")]
p = {j: lowpass(rd(j), apix, 10) for j in ["J2307", "J2384", "J2334", "J1049"] + [j for _, j in G]}
envp = envelope([p["J2307"], p["J2384"], p["J2334"], p["J1049"]], apix, None, k=1.0, dil=3) > 0.5
mean3 = np.mean([p[j][envp] for j in ("J2307", "J2384", "J2334")], 0); nm = lambda v: (lambda ab: ab[0] * v + ab[1])(np.polyfit(v, mean3, 1))
X = {j: nm(m[envp].astype(np.float64)) for j, m in p.items()}; un = (X["J2334"] - X["J2307"]) @ (X["J2334"] - X["J2307"])
d = lambda a, b: float((X[a] - X[b]) @ (X[a] - X[b]) / un)
print("pairwise map distance d (units of |e6-e8|^2); noise floor for two seeds of one class ~ 0.02-0.03")
print("%-7s" % "" + "".join("%7s" % n for n, _ in G))
for n, j in G:
    print("%-7s" % n + "".join("%7.3f" % d(j, j2) for _, j2 in G))
print("\nneighbour step d(size_i, size_i+1): " + "  ".join("%s>%s %.3f" % (G[i][0], G[i + 1][0], d(G[i][1], G[i + 1][1])) for i in range(len(G) - 1)))
print("distance of each size to the 60k rule pick J2384: " + "  ".join("%s %.3f" % (n, d(j, "J2384")) for n, j in G))
