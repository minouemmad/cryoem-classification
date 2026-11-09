"""NU-refinement poses -> NAME_particles_NUposes.cs in NAME_particles.cs (= cryoDRGN) order, matched by
(stack file, index). Particles NU dropped take their pose from the rejected set or the homogeneous refinement.
Usage: reorder_poses.py NAME
"""
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(__file__))
from registry import R, SETS  # noqa: E402

name = sys.argv[1]
N = Path(R) / name / "nu_consensus"
job = json.loads((N / "cryosparc_job.json").read_text())
pdir = SETS[name]["project_dir"]
ref = np.load(N / f"{name}_particles.cs")


def key(path, idx):
    b = os.path.basename(path.decode())
    if "/imported/" in path.decode():
        b = b.split("_", 1)[1]
    return b, int(idx)


pos = {key(p, i): k for k, (p, i) in enumerate(zip(ref["blob/path"], ref["blob/idx"]))}


def last(jdir, suffix):
    its = sorted(int(m.group(1)) for f in glob.glob(f"{jdir}/*_particles.cs") if (m := re.search(r"_(\d{3})_particles\.cs$", f)))
    if its:
        f = glob.glob(f"{jdir}/*_{its[-1]:03d}_particles{suffix}.cs")
        return (np.load(f[0]), its[-1]) if f else (None, its[-1])
    return None, None


nu, it = last(f"{pdir}/{job['nu']}", "")
rej, _ = last(f"{pdir}/{job['nu']}", "_rejected")
homo, hit = last(f"{pdir}/{job['homo']}", "")
hrej, _ = last(f"{pdir}/{job['homo']}", "_rejected")
fields = ["alignments3D/pose", "alignments3D/shift", "alignments3D/psize_A"]
pose_dt = [("alignments3D/pose", "<f4", (3,)), ("alignments3D/shift", "<f4", (2,)), ("alignments3D/psize_A", "<f4")]
keep = [n for n in ref.dtype.names if not n.startswith("alignments3D/")]
out = np.zeros(len(ref), dtype=[(n, ref.dtype[n]) for n in keep] + pose_dt)
for n in keep:
    out[n] = ref[n]
have = np.zeros(len(ref), bool)
counts = {}
for src, label in ((nu, "NU"), (rej, "NU rejected"), (homo, "homo refine"), (hrej, "homo rejected")):
    if src is None or len(src) == 0:
        continue
    rows = np.array([pos[key(p, i)] for p, i in zip(src["blob/path"], src["blob/idx"])], dtype=int)
    new = ~have[rows]
    for f in fields:
        out[f][rows[new]] = src[f][new]
    have[rows[new]] = True
    counts[label] = int(new.sum())
assert have.all(), f"{(~have).sum()} particles without a pose"
if homo is not None:
    hrows = np.array([pos[key(p, i)] for p, i in zip(homo["blob/path"], homo["blob/idx"])])
    hp = np.zeros((len(ref), 3)); hp[hrows] = homo["alignments3D/pose"]
    d = np.degrees((Rot.from_rotvec(out["alignments3D/pose"]) * Rot.from_rotvec(hp).inv()).magnitude())
    counts.update(pose_change_vs_homo_median_deg=float(np.median(d)), pose_change_vs_homo_p90_deg=float(np.percentile(d, 90)))
counts.update(nu_iteration=it, homo_iteration=hit)
with open(N / f"{name}_particles_NUposes.cs", "wb") as fh:
    np.save(fh, out)
print(json.dumps(counts))
