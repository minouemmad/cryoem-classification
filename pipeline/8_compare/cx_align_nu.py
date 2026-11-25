"""ChimeraX: place cryoSPARC NU maps (last iteration: map, half A, half B) in the deck frame ($DECK_REF): each map fitted on its own (ab initio maps can come out in either hand; both tried, best correlation
kept), the same placement applied to its two half maps, all resampled onto the 240 px reference grid -> OUT/<name>{,_hA,_hB}.mrc.
env SPEC = json {"out": dir, "maps": {name: [project_dir, job]}}. Writes OUT/fit.json. Existing outputs are skipped."""
import json, glob, os, re
from chimerax.core.commands import run
S = json.load(open(os.environ["SPEC"])); L = S["out"]; os.makedirs(L, exist_ok=True)
REF = os.environ["DECK_REF"]     # deck-frame reference map (J4001 deck: P25 J1497 class 8)
fitf = f"{L}/fit.json"; fits_out = json.load(open(fitf)) if os.path.exists(fitf) else {}
ref = run(session, f"open {REF}")[0]
ref_lp = run(session, f"volume gaussian #{ref.id_string} sDev 3")
run(session, f"volume #{ref_lp.id_string} encloseVolume 250000")


def opn(path, hand):
    m = run(session, f"open {path}")[0]
    if hand == "mirror":
        f = run(session, f"volume flip #{m.id_string} axis z"); run(session, f"close #{m.id_string}"); m = f
    return m


for name, (pdir, job) in S["maps"].items():
    if name in fits_out and os.path.exists(f"{L}/{name}_hB.mrc"):
        continue
    its = sorted(int(m.group(1)) for f in glob.glob(f"{pdir}/{job}/{job}_*_volume_map.mrc") if (m := re.search(r"_(\d{3})_volume_map\.mrc$", f)))
    pre = f"{pdir}/{job}/{job}_{its[-1]:03d}_volume_map"
    best = None
    for hand in ("same", "mirror"):
        m = opn(pre + ".mrc", hand)
        g = run(session, f"volume gaussian #{m.id_string} sDev 3")
        run(session, f"volume #{g.id_string} encloseVolume 250000 step 2")   # fit on every 2nd voxel (map is 3 A-smoothed)
        fits = run(session, f"fitmap #{g.id_string} inMap #{ref_lp.id_string} metric correlation search 60 placement sr radius 15 clusterAngle 6 clusterShift 3 logFits none")
        b = max(fits, key=lambda x: x.correlation()); b.place_models(session)
        cc = float(b.correlation())
        if best is None or cc > best[1]:
            best = (hand, cc, g.position)
        run(session, f"close #{m.id_string},{g.id_string}")
    hand, cc, pos = best
    for suf, out in (("", ""), ("_half_A", "_hA"), ("_half_B", "_hB")):
        m = opn(pre + suf + ".mrc", hand); m.position = pos
        rs = run(session, f"volume resample #{m.id_string} onGrid #{ref.id_string}")
        run(session, f"save {L}/{name}{out}.mrc models #{rs.id_string}")
        run(session, f"close #{m.id_string},{rs.id_string}")
    fits_out[name] = dict(job=job, file=pre + ".mrc", hand=hand, cc=cc)
    json.dump(fits_out, open(fitf, "w"), indent=1)
    print("SAVED", name, hand, round(cc, 3), flush=True)
