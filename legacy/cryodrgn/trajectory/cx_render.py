"""ChimeraX: render the masked trajectory maps at 35,000 A^3 with a fixed camera, surface coloured by the
difference from the trajectory mean (blue less, red more density). Env TRAJ = trajectory dir.
"""
import json, glob, os
from chimerax.core.commands import run
L = os.environ["TRAJ"]; V7 = "/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn/J4001_gP25_WT/candidate_groups_epoch50/presentation_v7"
R = json.load(open(f"{L}/maskedT/render.json")); x = 2.0 * R["diff_sd"]
os.makedirs(f"{L}/renders", exist_ok=True)
run(session, "set bgColor white"); run(session, "lighting full multiShadow 512"); run(session, "graphics silhouettes true width 1.2")
ref = run(session, f"open {V7}/maskedT/class_P8.mrc")[0]
run(session, f"volume #{ref.id_string} step 1"); run(session, f"volume #{ref.id_string} fastEncloseVolume 35000")
run(session, "view orient"); run(session, f"view #{ref.id_string}")
run(session, f"turn x 90 center #{ref.id_string}"); run(session, f"turn y 90 center #{ref.id_string}")
run(session, "zoom 0.8"); run(session, f"volume #{ref.id_string} hide")
names = [os.path.basename(f)[:-4] for f in sorted(glob.glob(f"{L}/maskedT/lat_0*.mrc")) + sorted(glob.glob(f"{L}/maskedT/vol_0*.mrc")) if "_diff" not in os.path.basename(f)]
for n in names:
    m = run(session, f"open {L}/maskedT/{n}.mrc")[0]
    run(session, f"volume #{m.id_string} step 1"); run(session, f"volume #{m.id_string} fastEncloseVolume 35000")
    run(session, f"surface dust #{m.id_string} size 6")
    d = run(session, f"open {L}/maskedT/{n}_diff.mrc")[0]; run(session, f"volume #{d.id_string} hide")
    run(session, f"color sample #{m.id_string} map #{d.id_string} palette {-x},#2C6FB7:0,#ECEDEE:{x},#C8382B")
    run(session, f"save {L}/renders/{n}.png width 700 height 700 supersample 3 transparentBackground true")
    print("RENDER", n, flush=True)
    run(session, f"close #{m.id_string},{d.id_string}")
