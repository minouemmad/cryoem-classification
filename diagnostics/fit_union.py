# ChimeraX script (11-25): fit 6MSM (NBD dimer, ATP) and 5UAK (dephosphorylated, NBDs apart) into consensus.mrc of the current env/<DS> dir. No heterorefinement info.
# 6MSM: global whole-model search, then local refit on its TMDs (TMD1 1-380, TMD2 840-1172) moving the whole model.  5UAK: superposed on the fitted 6MSM by its TMDs
# (matchmaker, whole model moves), with no refit (a local refit drifted on J2708), so its own NBD placement is kept relative to the membrane domains.  Hand from regions.log.  Saves fit_<model>.pdb.
import os, re
from chimerax.core.commands import run
K = os.getcwd(); E = os.environ.get("RES", "/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn") + "/model_pipeline/env/"; T = "/A:1-380,840-1172"
hand = int(re.search(r"hand (\d)", open("regions.log").read()).group(1)); vol = ("consensus.mrc", "consensusflip.mrc")[hand]
run(session, "close"); v = run(session, "open %s/%s" % (K, vol))[0]; a = run(session, "open %s6msm_protein.pdb" % E)[0]; b = run(session, "open %s5uak_protein.pdb" % E)[0]
g = run(session, "fitmap #%s inMap #%s resolution 10 search 400 radius 120 placement sr" % (a.id_string, v.id_string))
t1 = run(session, "fitmap #%s%s inMap #%s resolution 10 moveWholeMolecules true" % (a.id_string, T, v.id_string))
mm = run(session, "matchmaker #%s%s to #%s%s" % (b.id_string, T, a.id_string, T))
for m, nm in ((a, "6msm"), (b, "5uak")): run(session, "save %s/fit_%s.pdb models #%s relModel #%s" % (K, nm, m.id_string, v.id_string))
print("UNIONFIT hand", hand, "6msm global", round(g[0].correlation(), 3), "tmd", round(t1[0].correlation(), 3), 
      "| matchmaker rmsd", [round(x["final RMSD"], 2) for x in mm] if isinstance(mm, list) else mm, flush=True)
