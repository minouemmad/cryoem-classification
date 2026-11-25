"""ChimeraX: fit the mean decoded map onto the J1497 class-8 reference (both hands), apply that one placement to
every trajectory map and resample onto the reference grid -> $TRAJ/aligned/ (+ fit.json).
"""
import json, glob, os
from chimerax.core.commands import run
L = os.environ["TRAJ"]
REF = ("/data/hCFTR/mae2183/cryoem-classification/data/J4001_gP25_WT/heteroref_J1497/heteroref_volumes/"
       "cryosparc_P25_J1497_class_08_00000_volume_sharp.mrc")
os.makedirs(f"{L}/aligned", exist_ok=True)
ref = run(session, f"open {REF}")[0]
ref_lp = run(session, f"volume gaussian #{ref.id_string} sDev 3")
run(session, f"volume #{ref_lp.id_string} encloseVolume 250000")
def opn(path, hand):
    m = run(session, f"open {path}")[0]
    if hand == "mirror":
        f = run(session, f"volume flip #{m.id_string} axis z"); run(session, f"close #{m.id_string}"); m = f
    return m
best = None
for hand in ("same", "mirror"):
    m = opn(f"{L}/dec_mean.mrc", hand)
    g = run(session, f"volume gaussian #{m.id_string} sDev 3")
    run(session, f"volume #{g.id_string} encloseVolume 250000")
    fits = run(session, f"fitmap #{g.id_string} inMap #{ref_lp.id_string} metric correlation search 60 placement sr radius 15 clusterAngle 6 clusterShift 3 logFits none")
    b = max(fits, key=lambda x: x.correlation()); b.place_models(session)
    cc = float(b.correlation()); print("HAND", hand, cc, flush=True)
    if best is None or cc > best[1]:
        best = (hand, cc, g.position)
    run(session, f"close #{m.id_string},{g.id_string}")
hand, cc, pos = best
json.dump(dict(hand=hand, cc=cc), open(f"{L}/aligned/fit.json", "w"))
items = [(f"{d}_{os.path.basename(f)[4:7]}", f) for d in ("lat", "vol", "latpc") for f in sorted(glob.glob(f"{L}/dec_{d}/vol_*.mrc"))]
items += [("dec_mean", f"{L}/dec_mean.mrc")] + [(f"load_{os.path.basename(f)[:-4]}", f) for f in sorted(glob.glob(f"{L}/loadings/*.mrc"))]
for name, path in items:
    m = opn(path, hand); m.position = pos
    rs = run(session, f"volume resample #{m.id_string} onGrid #{ref.id_string}")
    run(session, f"save {L}/aligned/{name}.mrc models #{rs.id_string}")
    run(session, f"close #{m.id_string},{rs.id_string}")
    print("SAVED", name, flush=True)
