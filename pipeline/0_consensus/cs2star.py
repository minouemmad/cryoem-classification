"""cryoSPARC particle .cs (blob + ctf + alignments3D) -> RELION 3.1 .star.
Poses: rotvec -> intrinsic ZYZ Euler (rot, tilt, psi); origins = shift (px) x psize (A).
Usage: cs2star.py IN.cs OUT.star [--n-optics 36]
"""
import argparse
import numpy as np
from scipy.spatial.transform import Rotation

p = argparse.ArgumentParser()
p.add_argument("cs"); p.add_argument("star"); p.add_argument("--n-optics", type=int, default=36)
a = p.parse_args()
cs = np.load(a.cs)
c0 = cs[0]
ang = Rotation.from_rotvec(cs["alignments3D/pose"].astype(np.float64)).as_euler("ZYZ", degrees=True)
org = cs["alignments3D/shift"].astype(np.float64) * cs["alignments3D/psize_A"][:, None]
with open(a.star, "w") as f:
    f.write("\ndata_optics\n\nloop_\n")
    for i, k in enumerate(["rlnVoltage", "rlnImagePixelSize", "rlnSphericalAberration", "rlnAmplitudeContrast",
                           "rlnOpticsGroup", "rlnImageSize", "rlnImageDimensionality", "rlnOpticsGroupName"]):
        f.write(f"_{k} #{i + 1} \n")
    for g in range(1, a.n_optics + 1):
        f.write(f"{c0['ctf/accel_kv']:.6f} {c0['blob/psize_A']:.6f} {c0['ctf/cs_mm']:.6f} {c0['ctf/amp_contrast']:.6f} "
                f"{g} {c0['blob/shape'][0]} 2 opticsGroup{g}\n")
    f.write("\n\ndata_particles\n\nloop_\n")
    for i, k in enumerate(["rlnImageName", "rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi", "rlnOriginXAngst",
                           "rlnOriginYAngst", "rlnDefocusU", "rlnDefocusV", "rlnDefocusAngle", "rlnPhaseShift",
                           "rlnCtfBfactor", "rlnOpticsGroup", "rlnRandomSubset", "rlnClassNumber"]):
        f.write(f"_{k} #{i + 1} \n")
    for j, c in enumerate(cs):
        f.write(f"{c['blob/idx'] + 1:06d}@{c['blob/path'].decode()} {ang[j, 0]:.6f} {ang[j, 1]:.6f} {ang[j, 2]:.6f} "
                f"{org[j, 0]:.6f} {org[j, 1]:.6f} {c['ctf/df1_A']:.6f} {c['ctf/df2_A']:.6f} "
                f"{np.degrees(c['ctf/df_angle_rad']):.6f} {np.degrees(c['ctf/phase_shift_rad']):.6f} {c['ctf/bfactor']:.6f} "
                f"{c['ctf/exp_group_id'] + 1} {c['alignments3D/split'] + 1} 1\n")
    f.write("\n")
