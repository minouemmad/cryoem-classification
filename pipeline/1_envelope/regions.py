"""Domain regions TMD1 / NBD1 / TMD2 / NBD2 and the PDB envelope from the best fit in fit_out.txt (run in the envelope dir).
env64_pdb.npy = protein region of the fitted model dilated by 2 voxels (the envelope used by default). A hand-1 fit means the map was
flipped, so the regions are flipped back to the pipeline frame.   -> regions.npz, env64_pdb.npy"""
import json, os, re
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation
A = float(open("apix.txt").read()); N = 64; txt = open("fit_out.txt").read(); P = os.environ["ROOT"] + "/reference/pdb/"
fits = [(mo, int(h), json.loads(c)[0], np.array(json.loads(p))) for mo, h, c, p in re.findall(r"MODEL (\S+) HAND (\d) CORR (\[.*?\]) POS (\[\[.*?\]\]) MPOS", txt)]
model, hand, corr, T = max(fits, key=lambda x: x[2])
r = np.array([(int(l[22:26]), float(l[30:38]), float(l[38:46]), float(l[46:54])) for l in open(P + {"vshape": "D1557_fitT.pdb", "6msm": "6msm_protein.pdb"}[model]) if l.startswith("ATOM")])
xyz = (T[:, :3] @ r[:, 1:].T).T + T[:, 3]
DOM = {"TMD1": [(1, 380)], "NBD1": [(381, 645)], "TMD2": [(840, 1172)], "NBD2": [(1173, 1480)]}


def dens(sel):
    g = np.zeros((N, N, N), np.float32); idx = np.rint(xyz[sel] / A).astype(int); idx = idx[((idx >= 0) & (idx < N)).all(1)]
    np.add.at(g, (idx[:, 2], idx[:, 1], idx[:, 0]), 1.0); return gaussian_filter(g, 1.5)


allm = dens(np.ones(len(r), bool)); m = np.load("consensus64.npy"); m = m[::-1] if hand else m; mk = allm > allm.max() * 0.05
print("best", model, "hand", hand, "corr", corr, "| model-vs-consensus corr", round(float(np.corrcoef(allm[mk], m[mk])[0, 1]), 3))
stack = np.stack([dens(np.any([(r[:, 0] >= a) & (r[:, 0] <= b) for a, b in rs], 0)) for rs in DOM.values()])
prot = allm > allm.max() * 0.10; lab = stack.argmax(0)
out = {k: prot & (lab == i) & (stack[i] > 0.02) for i, k in enumerate(DOM)}
if hand: out = {k: v[::-1].copy() for k, v in out.items()}; prot = prot[::-1].copy()
np.savez("regions.npz", **out, PROT=prot); e = binary_dilation(prot, iterations=2); np.save("env64_pdb.npy", e)
c = np.load("env64.npy"); print("ENVPDB", int(e.sum()), "| consensus envelope", int(c.sum()), "| overlap", int((e & c).sum()), "| " + " ".join(f"{k} {int(v.sum())}" for k, v in out.items()))
