"""ChimeraX: refine the rigid fit of each moving map onto its reference (local fitmap) and resample it on the
reference grid, so difference maps show density changes rather than misregistration.
Env PAIRS = "out.mrc=moving.mrc,ref.mrc;..."; optional COREPDB (+ CORERES) restricts the fit to that zone.
Run: chimerax --nogui --offscreen --exit --script cx_pairfit.py
"""
import os
from chimerax.core.commands import run
for item in os.environ["PAIRS"].split(";"):
    if not item.strip():
        continue
    out, rest = item.split("="); mov, ref = rest.split(",")
    r = run(session, f"open {ref}")[0]; m = run(session, f"open {mov}")[0]
    p0 = m.position
    core = os.environ.get("COREPDB")            # fit on the TMD core only (CFTR's rigid frame); NBD motions stay in the difference
    if core:
        mol = run(session, f"open {core}")[0]
        z = run(session, f"volume zone #{m.id_string} nearAtoms #{mol.id_string}:{os.environ.get('CORERES', '1-388,841-1173')} range 6 newMap true")
        fits = run(session, f"fitmap #{z.id_string} inMap #{r.id_string} metric correlation envelope false shift true rotate true")
        m.position = z.position
    else:
        fits = run(session, f"fitmap #{m.id_string} inMap #{r.id_string} metric correlation envelope false shift true rotate true")
    f = fits[0] if isinstance(fits, (list, tuple)) else fits
    d = m.position * p0.inverse(); ax, ang = d.rotation_axis_and_angle(); sh = float(sum(x * x for x in d.translation()) ** 0.5)
    rs = run(session, f"volume resample #{m.id_string} onGrid #{r.id_string}")
    run(session, f"save {out} models #{rs.id_string}")
    print("FIT", out, round(float(f.correlation()), 4), round(sh, 2), round(float(ang), 2), flush=True)
    run(session, "close")
