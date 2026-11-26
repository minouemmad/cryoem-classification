"""(11-25) Evaluate the J4001 per-domain residual models (label-free fits; heterorefinement labels J1497 used ONLY to score):
reproducibility (fold-vs-fold subspace cosines), redundancy with the whole-molecule null model (canonical correlations), and how well
domain coordinates separate P7 / P8 / P6 particles (single-coordinate AUC, 5-fold cross-validated logistic AUC) versus the whole-molecule coordinates."""
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.cross_decomposition import CCA
R = os.environ.get("DATA_ROOT", "/data/hCFTR/mae2183/cryoem-classification") + "/"; P = R + "results_cryodrgn/model_pipeline/pipe/J4001/"
uid = np.load(R + "data/J4001_gP25_WT/homoref_J4001/cryosparc_P25_J4001_ctf_particles.cs")["uid"]; pos = {u: i for i, u in enumerate(uid)}; lab = np.full(len(uid), -1)
for k in range(6, 11):
    lab[[pos[x] for x in np.load(R + f"data/J4001_gP25_WT/heteroref_J1497/cryosparc_P25_J1497_class_{k:02d}_00000_particles.cs")["uid"] if x in pos]] = k
G = np.load(P + "start_free/start/ppca_null_z.npy"); Z = {}
for d in ("TMD1", "NBD1", "TMD2", "NBD2"):
    Z[d] = np.load(P + f"domain_fit/{d}/ppca_dom_z.npy"); fc = [l for l in open(P + f"domain_fit/{d}/ppca_dom.txt") if "fold-vs-fold subspace cosines" in l][-1].split(":")[-1].split()
    cc = CCA(n_components=2).fit(Z[d], G); a, b = cc.transform(Z[d], G)
    print(f"{d}: fold cosines {' '.join(fc)} | canonical corr with the 8 whole-molecule coords {np.corrcoef(a[:, 0], b[:, 0])[0, 1]:.2f} {np.corrcoef(a[:, 1], b[:, 1])[0, 1]:.2f}")
Dall = np.hstack([Z[d] for d in Z]); rng = np.random.default_rng(0); sub = rng.choice(len(uid), 80000, replace=False)
def cvauc(X, y):
    X = (X - X.mean(0)) / X.std(0); p = cross_val_predict(LogisticRegression(max_iter=2000), X[sub], y[sub], cv=5, method="predict_proba")[:, 1]; return roc_auc_score(y[sub], p)
for nm, y in (("P7 vs rest", lab == 7), ("P8 vs rest", lab == 8), ("P6 vs rest", lab == 6)):
    single = {d: max(max(roc_auc_score(y, Z[d][:, j]), 1 - roc_auc_score(y, Z[d][:, j])) for j in range(Z[d].shape[1])) for d in Z}
    print(f"{nm}: best single domain coord AUC", {d: round(v, 2) for d, v in single.items()},
          f"| CV AUC whole-molecule 8: {cvauc(G, y):.3f}  domains 16: {cvauc(Dall, y):.3f}  both 24: {cvauc(np.hstack([G, Dall]), y):.3f}")
