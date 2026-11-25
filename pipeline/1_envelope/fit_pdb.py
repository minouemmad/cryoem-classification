# ChimeraX: fit each CFTR model (reference/pdb) into consensus.mrc of the current dir, both hands; prints MODEL / HAND / CORR / transforms.
#   vshape = D1557_fitT.pdb (inward-facing, NBDs apart; best for J4001 / J4624);  6msm = human CFTR E1371Q, ATP-bound NBD dimer (best for J207 / J2708 / J399).
import os
from chimerax.core.commands import run
P = os.environ["ROOT"] + "/reference/pdb/"; AP = float(open("apix.txt").read())
for name, pdb in {"vshape": P + "D1557_fitT.pdb", "6msm": P + "6msm_protein.pdb"}.items():
    for hand, vol in ((0, "consensus.mrc"), (1, "consensusflip.mrc")):
        run(session, "close"); v = run(session, f"open {os.getcwd()}/{vol}")[0]; m = run(session, "open " + pdb)[0]
        mm = run(session, "molmap #%s 10 gridSpacing %.4f" % (m.id_string, AP))
        r = run(session, "fitmap #%s inMap #%s search 400 radius 120 placement sr moveWholeMolecules true" % (mm.id_string, v.id_string))
        print("MODEL", name, "HAND", hand, "CORR", [round(f.correlation(), 3) for f in r[:4]], "POS", mm.scene_position.matrix.tolist(), "MPOS", m.scene_position.matrix.tolist(), flush=True)
