"""ChimeraX: align plain map files (no half maps). env SPEC = json {"out": dir, "maps": {name: path}}."""
import json, os
from chimerax.core.commands import run
S = json.load(open(os.environ["SPEC"])); L = S["out"]; os.makedirs(L, exist_ok=True)
REF = ("/data/hCFTR/mae2183/cryoem-classification/data/gP25W6_J1442_J1497/J1497_classes/heteroref_volumes/cryosparc_P25_J1497_class_08_00000_volume_sharp.mrc")
ref = run(session, f"open {REF}")[0]; ref_lp = run(session, f"volume gaussian #{ref.id_string} sDev 3"); run(session, f"volume #{ref_lp.id_string} encloseVolume 250000")
def opn(path, hand):
    m = run(session, f"open {path}")[0]
    if hand == "mirror":
        f = run(session, f"volume flip #{m.id_string} axis z"); run(session, f"close #{m.id_string}"); m = f
    return m
for name, path in S["maps"].items():
    if os.path.exists(f"{L}/{name}.mrc"): continue
    best = None
    for hand in ("same", "mirror"):
        m = opn(path, hand); g = run(session, f"volume gaussian #{m.id_string} sDev 3"); run(session, f"volume #{g.id_string} encloseVolume 250000 step 2")
        fits = run(session, f"fitmap #{g.id_string} inMap #{ref_lp.id_string} metric correlation search 60 placement sr radius 15 clusterAngle 6 clusterShift 3 logFits none")
        b = max(fits, key=lambda x: x.correlation()); b.place_models(session); cc = float(b.correlation())
        if best is None or cc > best[1]: best = (hand, cc, g.position)
        run(session, f"close #{m.id_string},{g.id_string}")
    hand, cc, pos = best; m = opn(path, hand); m.position = pos; rs = run(session, f"volume resample #{m.id_string} onGrid #{ref.id_string}")
    run(session, f"save {L}/{name}.mrc models #{rs.id_string}"); run(session, f"close #{m.id_string},{rs.id_string}"); print("SAVED", name, hand, round(cc, 3), flush=True)
