"""(11-25) Why is there no P7 map among the J4001 start-axis classes?  (evaluation only: heterorefinement maps / labels are used to score, never to build anything)
A. maps (10 A fits from od_compare, J4001 protein mask): raw corr of the reference maps, P7 as a blend of P8/P6, d2 to P7 (|x-P7|^2/|P8-P6|^2) of every class map + consensus.
B. particles (J1497 heterorefinement: class 06/07/08 = P6/P7/P8, joined by uid to the consensus .cs = stack order): where P7 particles go in each K set,
   and how well each start / search coordinate separates P7 particles from the rest (AUC)."""
import os
import glob, json, numpy as np, mrcfile
from sklearn.metrics import roc_auc_score
R = os.environ.get("DATA_ROOT", "/data/hCFTR/mae2183/cryoem-classification") + "/"; RC = R + "results_cryodrgn/"; MR = RC + "J4001_gP25_WT/method_rank/fits/"; OD = RC + "model_pipeline/od/"
mask = mrcfile.read(RC + "J4001_gP25_WT/candidate_groups_epoch50/presentation_v7/models/protein_maskT.mrc") > 0.5
rd = lambda f: mrcfile.read(f)[mask].astype(np.float64)
ref = {k: rd(MR + f) for k, f in (("P8", "fit_J2975.mrc"), ("P7", "fit_J2974.mrc"), ("P6", "fit_J2973.mrc"), ("Alt0", "fit_Alt0.mrc"), ("Alt1", "fit_Alt1.mrc"), ("cons", "in_J1049.mrc"))}
d86 = ((ref["P8"] - ref["P6"]) ** 2).sum(); d2 = lambda x, y: float(((x - y) ** 2).sum() / d86)
print("A. raw corr:", {f"{a}-{b}": round(float(np.corrcoef(ref[a], ref[b])[0, 1]), 3) for a, b in (("P8", "P6"), ("P7", "P8"), ("P7", "P6"), ("P7", "cons"), ("P8", "cons"), ("P6", "cons"))})
print("   d2 to P7:", {k: round(d2(v, ref["P7"]), 3) for k, v in ref.items() if k != "P7"})
X = np.stack([ref["P8"], ref["P6"], np.ones_like(ref["P8"])], 1); c, *_ = np.linalg.lstsq(X, ref["P7"], rcond=None); res = ref["P7"] - X @ c
print(f"   P7 ~ {c[0]:.2f} P8 + {c[1]:.2f} P6: unexplained |res|^2/d86 = {(res ** 2).sum() / d86:.3f} (best-possible blend map is still this far from P7)")
d = json.load(open(OD + "cmp/compare_J4001.json")); best = {}
for r in d["rows"]:
    if r["run"] == "hetero": continue
    x = rd(r["path"]); v = (d2(x, ref["P7"]), r["name"], r["n"], round(r["cc_diff"]["P7"], 2), r["closest"]); best.setdefault(r["run"], []).append(v)
allv = sorted(v for l in best.values() for v in l)
print("   closest class maps to P7 (d2, name, n, cc_diff P7, closest ref):"); [print("     %.3f %-24s %6d %5.2f %s" % v) for v in allv[:8]]
print("   per start best d2 to P7:", {k: round(min(l)[0], 3) for k, l in best.items()}, "| maps whose closest ref is P7:", sum(v[4] == "P7" for v in allv), "of", len(allv))
# B. particles
import sys; sys.path.insert(0, "/home/mae2183/miniconda3/envs/cryodrgn/lib/python3.10/site-packages")
uid = np.load(R + "data/J4001_gP25_WT/homoref_J4001/cryosparc_P25_J4001_ctf_particles.cs")["uid"]; pos = {u: i for i, u in enumerate(uid)}
lab = np.full(len(uid), -1)
for k, nm in ((6, "P6"), (7, "P7"), (8, "P8"), (9, "P9"), (10, "P10")):
    u = np.load(R + f"data/J4001_gP25_WT/heteroref_J1497/cryosparc_P25_J1497_class_{k:02d}_00000_particles.cs")["uid"]; ii = [pos[x] for x in u if x in pos]; lab[ii] = k
has = lab >= 0; p7 = lab == 7
print(f"\nB. J1497 labels joined: {has.sum()} of {len(uid)} | class sizes", {k: int((lab == k).sum()) for k in range(6, 11)})
for run in ("start_free", "start_vq", "start_mlp", "start_diffmap", "start_latent"):
    for K in (3, 4, 5):
        f = OD + f"J4001/{run}_K{K}/labels.npy"
        try: L = np.load(f)
        except Exception: continue
        print(f"  {run} K{K}: P7 particles per class", [round(float((p7 & (L == k)).sum() / p7.sum()), 2) for k in range(K)],
              "| P7 share of each class", [round(float((p7 & (L == k)).sum() / (has & (L == k)).sum()), 2) for k in range(K)], f"(overall {p7.sum() / has.sum():.2f})")
    P = RC + f"model_pipeline/pipe/J4001/{run}/"
    try:
        S = np.load(P + "iter2/S_final.npy"); s0 = np.load(P + "pc.scores.npy")[:, 0]
        au = lambda v: round(max(roc_auc_score(p7[has], v[has]), 1 - roc_auc_score(p7[has], v[has])), 2)
        mid = np.abs(s0 - np.median(s0)) / (np.abs(s0 - np.median(s0)).max())
        print(f"  {run}: AUC P7-vs-rest start axis {au(s0)}, |start - median| {au(-mid)}, search coords", [au(S[:, j]) for j in range(min(6, S.shape[1]))])
    except Exception as e: print(" ", run, "no coords", e)
