"""ChimeraX: render difference maps (mapdiff.py) as red (+) / blue (-) surfaces over a transparent reference,
one shared threshold and camera for every panel.
Env SPEC = json {"camera", "camera_level", "turn", "zoom", "thr", "outdir", "panels": [{"name", "ref", "ref_level", "diff"}]}
Run: chimerax --nogui --offscreen --exit --script cx_render_mapdiff.py
"""
import json, os
from chimerax.core.commands import run
S = json.load(open(os.environ["SPEC"]))
run(session, "set bgColor white"); run(session, "lighting soft"); run(session, "graphics silhouettes true width 1.2")
cam = run(session, f"open {S['camera']}")[0]
run(session, f"volume #{cam.id_string} level {S['camera_level']} step 1"); run(session, "view orient"); run(session, f"view #{cam.id_string}")
for t in S.get("turn", []):
    run(session, f"turn {t} center #{cam.id_string}")
run(session, f"zoom {S.get('zoom', 0.85)}"); run(session, f"volume #{cam.id_string} hide")
t = S["thr"]
for p in S["panels"]:
    ids = []
    if p.get("ref"):
        r = run(session, f"open {p['ref']}")[0]; ids.append(r.id_string)
        run(session, f"volume #{r.id_string} level {p['ref_level']} step 1 color #b8bcc2 transparency 0.75")
    if p.get("diff"):
        d = run(session, f"open {p['diff']}")[0]; ids.append(d.id_string)
        n = run(session, f"volume scale #{d.id_string} factor -1"); ids.append(n.id_string)       # negative side as its own map
        run(session, f"volume #{d.id_string} show style surface level {t} color #d0362f step 1")
        run(session, f"volume #{n.id_string} style surface level {t} color #2f6fd0 step 1")
        run(session, f"surface dust #{d.id_string},{n.id_string} size {S.get('dust', 8)}")
    run(session, f"save {S['outdir']}/{p['name']}.png width {S.get('px', 600)} height {S.get('px', 600)} supersample 3 transparentBackground true")
    run(session, "close #" + ",".join(ids))
    print("RENDER", p["name"], flush=True)
