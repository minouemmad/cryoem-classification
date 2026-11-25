"""NAME_particles.cs -> NAME_particles.star (RELION 3.1, no poses: the homogeneous refinement searches from scratch).
One optics group per cryoSPARC exposure group; absolute stack paths; (stack name, index) checked unique so poses can be
matched back after import. With TREE_DIR TAG: one STAR per recon-tree leaf (leaves.npy, cryoDRGN order) instead,
-> TREE_DIR/cryosparc/TAG_leafK.star.
Usage: write_star.py NAME [TREE_DIR TAG]"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from registry import R  # noqa: E402


def write_full(name, N):
    cs = np.load(N / f"{name}_particles.cs")
    keys = list(zip([os.path.basename(p.decode()) for p in cs["blob/path"]], cs["blob/idx"].tolist()))
    assert len(set(keys)) == len(keys), "(stack basename, idx) not unique"
    grp = cs["ctf/exp_group_id"].astype(int)
    ug = np.unique(grp)
    gmap = {g: i + 1 for i, g in enumerate(ug)}
    with open(N / f"{name}_particles.star", "w") as f:
        f.write("\ndata_optics\n\nloop_\n")
        for i, k in enumerate(["rlnVoltage", "rlnImagePixelSize", "rlnSphericalAberration", "rlnAmplitudeContrast",
                               "rlnOpticsGroup", "rlnImageSize", "rlnImageDimensionality", "rlnOpticsGroupName"]):
            f.write(f"_{k} #{i + 1} \n")
        for g in ug:
            r = cs[grp == g][0]
            f.write(f"{r['ctf/accel_kv']:.6f} {r['blob/psize_A']:.6f} {r['ctf/cs_mm']:.6f} {r['ctf/amp_contrast']:.6f} "
                    f"{gmap[g]} {r['blob/shape'][0]} 2 opticsGroup{gmap[g]}\n")
        f.write("\n\ndata_particles\n\nloop_\n")
        for i, k in enumerate(["rlnImageName", "rlnDefocusU", "rlnDefocusV", "rlnDefocusAngle", "rlnPhaseShift",
                               "rlnCtfBfactor", "rlnOpticsGroup"]):
            f.write(f"_{k} #{i + 1} \n")
        for c, g in zip(cs, grp):
            f.write(f"{c['blob/idx'] + 1:06d}@{c['blob/path'].decode()} {c['ctf/df1_A']:.6f} {c['ctf/df2_A']:.6f} "
                    f"{np.degrees(c['ctf/df_angle_rad']):.6f} {np.degrees(c['ctf/phase_shift_rad']):.6f} {c['ctf/bfactor']:.6f} "
                    f"{gmap[g]}\n")
        f.write("\n")
    print(name, len(cs), "particles,", len(ug), "optics groups ->", N / f"{name}_particles.star")


def write_leaves(name, N, tree, tag):
    full = (N / f"{name}_particles.star").read_text().split("\n")
    hdr_end = next(i for i, l in enumerate(full) if l.startswith("_rlnOpticsGroup #7")) + 1
    rows = [l for l in full[hdr_end:] if l.strip()]
    leaves = np.load(tree / "leaves.npy")
    assert len(rows) == len(leaves), (len(rows), len(leaves))
    (tree / "cryosparc").mkdir(exist_ok=True)
    for k in range(leaves.max() + 1):
        out = tree / "cryosparc" / f"{tag}_leaf{k}.star"
        out.write_text("\n".join(full[:hdr_end] + [r for r, l in zip(rows, leaves) if l == k]) + "\n\n")
        print(out, int((leaves == k).sum()))


if __name__ == "__main__":
    name = sys.argv[1]
    N = Path(R) / name / "nu_consensus"
    if len(sys.argv) == 4:
        if not (N / f"{name}_particles.star").exists():
            write_full(name, N)
        write_leaves(name, N, Path(sys.argv[2]), sys.argv[3])
    else:
        write_full(name, N)
