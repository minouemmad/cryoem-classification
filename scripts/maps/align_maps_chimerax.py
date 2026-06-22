"""Rigidly align cryo-EM volumes into a common reference frame, then save them
resampled onto the reference grid so they can be compared voxel-by-voxel.

Why this exists
---------------
Independently-run refinements (e.g. separate NU jobs per class) each re-optimise
the *global* pose, so the output volumes can sit in different orientations /
origins inside the box. A direct voxel comparison (compare_maps.py) then reports
near-zero correlation that reflects misalignment, NOT real structural difference.
This script fixes that by:

  1. Opening every map in ChimeraX.
  2. fitmap <map> inMap <reference> search N   -> global rigid-body alignment.
  3. volume resample <map> onGrid <reference>  -> bake the transform into voxels.
  4. Saving each resampled map (reference saved as-is).

Output maps all share the reference grid/frame and are safe to feed to
compare_maps.py.

Usage
-----
python scripts/align_maps_chimerax.py \
    --maps data/maps/J1442_nu_r09/J3361_P6_n13211.mrc ... \
    --ref-index 1 \
    --search 100 \
    --outdir data/maps/J1442_nu_r09_aligned
"""

import argparse
import subprocess
import sys
from pathlib import Path

CHIMERAX = r"C:\Program Files\ChimeraX 1.12rc202606031852\bin\ChimeraX.exe"


def build_cxc(maps, ref_index, search, outdir, bin_spacing=None):
    """Return ChimeraX command-script text. ref_index is 1-based into `maps`.

    If bin_spacing is given (Angstrom/voxel), every map is first resampled to that
    coarser spacing; alignment + the saved output are done on the binned grid. This
    is ~ (bin/orig)^3 faster for the global fitmap search and preserves all
    conformationally relevant (> ~5 A) information.
    """
    lines = []
    for m in maps:
        lines.append(f'open "{Path(m).resolve().as_posix()}"')
    ref_model = ref_index  # ChimeraX assigns model #1.. in open order
    outdir = Path(outdir).resolve()

    if bin_spacing is not None:
        # Make binned copies #(10+i); ref binned copy is #(10+ref_model)
        for i in range(1, len(maps) + 1):
            lines.append(f'volume resample #{i} spacing {bin_spacing} modelId #{10 + i}')
        work_ref = 10 + ref_model
        for i, m in enumerate(maps, start=1):
            out = (outdir / Path(m).name).as_posix()
            work = 10 + i
            if i == ref_model:
                lines.append(f'save "{out}" model #{work}')
            else:
                lines.append(
                    f'fitmap #{work} inMap #{work_ref} search {search} metric correlation'
                )
                new_id = 100 + i
                lines.append(f'volume resample #{work} onGrid #{work_ref} modelId #{new_id}')
                lines.append(f'save "{out}" model #{new_id}')
    else:
        for i, m in enumerate(maps, start=1):
            out = (outdir / Path(m).name).as_posix()
            if i == ref_model:
                # reference defines the frame; save unchanged
                lines.append(f'save "{out}" model #{i}')
            else:
                lines.append(
                    f'fitmap #{i} inMap #{ref_model} search {search} metric correlation'
                )
                # resample the (now moved) map onto the reference grid -> new model
                new_id = 100 + i
                lines.append(f'volume resample #{i} onGrid #{ref_model} modelId #{new_id}')
                lines.append(f'save "{out}" model #{new_id}')
    lines.append("exit")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--maps", nargs="+", required=True)
    ap.add_argument("--ref-index", type=int, default=1,
                    help="1-based index into --maps used as the alignment reference")
    ap.add_argument("--search", type=int, default=100,
                    help="number of random initial orientations for global fitmap search")
    ap.add_argument("--bin", type=float, default=None,
                    help="resample to this voxel spacing (A) before aligning; "
                         "speeds up global search by (bin/orig)^3 and is fine for "
                         "conformational-scale (>~5 A) comparison")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--chimerax", default=CHIMERAX)
    args = ap.parse_args()

    missing = [m for m in args.maps if not Path(m).exists()]
    if missing:
        print("ERROR: missing input map(s):")
        for m in missing:
            print(f"  - {m}")
        sys.exit(2)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cxc_text = build_cxc(args.maps, args.ref_index, args.search, outdir, args.bin)
    cxc_path = outdir / "_align.cxc"
    cxc_path.write_text(cxc_text)
    print(f"Wrote ChimeraX script: {cxc_path}")
    print("-" * 60)
    print(cxc_text)
    print("-" * 60)

    cmd = [args.chimerax, "--nogui", "--offscreen", "--exit", str(cxc_path)]
    print("Running:", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:\n", proc.stdout[-4000:])
    if proc.stderr.strip():
        print("STDERR:\n", proc.stderr[-4000:])
    # ChimeraX sometimes prints command errors but still exits 0; treat those as failures.
    if proc.returncode != 0 or "ERROR:" in proc.stdout:
        if proc.returncode == 0:
            print("ERROR: ChimeraX reported command errors (see STDOUT).")
        sys.exit(proc.returncode)
    print(f"\nAligned maps written to {outdir.resolve()}")


if __name__ == "__main__":
    main()
