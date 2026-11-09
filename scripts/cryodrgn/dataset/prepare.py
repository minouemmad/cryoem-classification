"""Prepare one dataset for cryoSPARC import + cryoDRGN.
Writes results_cryodrgn/NAME/nu_consensus/: NAME_particles.cs (protein classes, absolute stack paths),
classes/<C>.cs (uids per heteroref class, evaluation only), init_volume.mrc (resampled to the particle grid), prep.json.
Usage: prepare.py NAME
"""
import json
import os
import sys
from pathlib import Path

import mrcfile
import numpy as np
import numpy.lib.recfunctions as rfn

sys.path.insert(0, os.path.dirname(__file__))
from registry import FIRST_PROTEIN_CLASS, R, SETS  # noqa: E402

name = sys.argv[1]
cfg = SETS[name]
out = Path(R) / name / "nu_consensus"
(out / "classes").mkdir(parents=True, exist_ok=True)

multi = np.load(cfg["multi"])
cls = np.argmax(multi["alignments3D_multi/class_posterior"], 1)
uid = multi["uid"]
parts = [np.load(f) for f in cfg["passthrough"]]
for p in parts:
    assert np.array_equal(p["uid"], uid), "passthrough / multi uid order differs"
src = {"uid": parts[0]}
for p in parts:
    for f in p.dtype.names:
        if f not in src and f.split("/")[0] in ("blob", "ctf", "location", "alignments3D"):
            src[f] = p
cs = np.empty(len(uid), dtype=[(f, p.dtype[f]) for f, p in src.items()])
for f, p in src.items():
    cs[f] = p[f]

keep = cls >= FIRST_PROTEIN_CLASS
cs = cs[keep]
cls_k = cls[keep]
root = cfg["stack_root"]
paths = np.array([os.path.join(root, p.decode().lstrip(">")) for p in cs["blob/path"]])
missing = [p for p in sorted(set(paths.tolist())) if not os.path.exists(p)]
assert not missing, f"{len(missing)} stacks missing, e.g. {missing[:2]}"
width = max(len(p) for p in paths) + 1
cs = rfn.drop_fields(cs, "blob/path", usemask=False)
cs = rfn.append_fields(cs, "blob/path", paths.astype(f"S{width}"), usemask=False)
cs = rfn.repack_fields(cs)
with open(out / f"{name}_particles.cs", "wb") as fh:
    np.save(fh, cs)

class_files = []
for c in range(FIRST_PROTEIN_CLASS, cls.max() + 1):
    u = np.ascontiguousarray(cs["uid"][cls_k == c])
    if len(u) == 0:
        continue
    f = out / "classes" / f"{cfg['heteroref']}_class_{c:02d}.cs"
    with open(f, "wb") as fh:
        np.save(fh, np.rec.fromarrays([u], names="uid"))
    class_files.append((f"C{c:02d}", str(f), int(len(u))))

box = int(cs["blob/shape"][0][0])
apix = float(cs["blob/psize_A"][0])

# starting map: Fourier crop/pad to the particle grid
with mrcfile.open(cfg["init_volume"], permissive=True) as m:
    v = m.data.astype(np.float32)
    va = float(m.voxel_size.x)
nv = v.shape[0]
target_n = int(round(nv * va / apix))          # same physical extent at the particle pixel size
F = np.fft.fftshift(np.fft.fftn(v))
if target_n <= nv:
    s = (nv - target_n) // 2
    F = F[s:s + target_n, s:s + target_n, s:s + target_n]
else:
    p0 = (target_n - nv) // 2
    F = np.pad(F, [(p0, target_n - nv - p0)] * 3)
w = np.real(np.fft.ifftn(np.fft.ifftshift(F))).astype(np.float32) * (target_n / nv) ** 3
if target_n != box:                              # then real-space crop / pad to the particle box
    if target_n > box:
        s = (target_n - box) // 2
        w = w[s:s + box, s:s + box, s:s + box]
    else:
        p0 = (box - target_n) // 2
        w = np.pad(w, [(p0, box - target_n - p0)] * 3)
with mrcfile.new(out / "init_volume.mrc", overwrite=True) as m:
    m.set_data(np.ascontiguousarray(w))
    m.voxel_size = apix

info = dict(name=name, n_heteroref=int(len(uid)), n_protein=int(keep.sum()),
            class_counts={c: n for c, _, n in class_files}, class_files=[f for _, f, _ in class_files],
            box=box, apix=apix, apix_128=apix * box / 128, n_stacks=int(len(set(paths.tolist()))),
            init_volume_src=cfg["init_volume"], init_volume_src_box=nv, init_volume_src_apix=va)
(out / "prep.json").write_text(json.dumps(info, indent=1))
print(json.dumps(info, indent=1))
