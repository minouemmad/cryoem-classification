"""ChimeraX: render maskedT/<name>.mrc for env NAMES (space separated) in the deck style (camera on $DECK/maskedT/class_P8, 35,000 A^3, dust < 6 A, surface coloured by <name>_diff, blue less /
red more density than the reference map). env D = dir holding maskedT/ -> D/renders/<name>.png"""
import json, os
from chimerax.core.commands import run
L = os.environ["D"]; V7 = os.environ["DECK"]
R = json.load(open(f"{L}/maskedT/render.json")); x = 2.0 * R["diff_sd"]
SOLID = dict(a.split("=") for a in os.environ.get("SOLID", "").split())
os.makedirs(f"{L}/renders", exist_ok=True)
run(session, "set bgColor white"); run(session, "lighting full multiShadow 512"); run(session, "graphics silhouettes true width 1.2")
ref = run(session, f"open {V7}/maskedT/class_P8.mrc")[0]
run(session, f"volume #{ref.id_string} step 1"); run(session, f"volume #{ref.id_string} fastEncloseVolume 35000")
run(session, "view orient"); run(session, f"view #{ref.id_string}")
run(session, f"turn x 90 center #{ref.id_string}"); run(session, f"turn y 90 center #{ref.id_string}")
run(session, "zoom 0.8"); run(session, f"volume #{ref.id_string} hide")
for n in os.environ["NAMES"].split():
    m = run(session, f"open {L}/maskedT/{n}.mrc")[0]
    run(session, f"volume #{m.id_string} step 1"); run(session, f"volume #{m.id_string} fastEncloseVolume 35000")
    run(session, f"surface dust #{m.id_string} size 6")
    if n in SOLID:                                   # env SOLID="name=#hex ...": plain colour
        run(session, f"volume #{m.id_string} color {SOLID[n]}")
        run(session, f"save {L}/renders/solid_{n}.png width 700 height 700 supersample 3 transparentBackground true")
        print("RENDER", n, flush=True); run(session, f"close #{m.id_string}"); continue
    d = run(session, f"open {L}/maskedT/{n}_diff.mrc")[0]; run(session, f"volume #{d.id_string} hide")
    run(session, f"color sample #{m.id_string} map #{d.id_string} palette {-x},#2C6FB7:0,#ECEDEE:{x},#C8382B")
    run(session, f"save {L}/renders/{n}.png width 700 height 700 supersample 3 transparentBackground true")
    print("RENDER", n, flush=True)
    run(session, f"close #{m.id_string},{d.id_string}")
