"""(11-25) per run: half-set agreement of the residual models (initial = unconditioned, second = conditioned on the start bins), sharpening gain from the residual
coordinates, and redundancy of the fully conditioned search coordinates.  Run on hudson in model_pipeline/."""
import glob, json, os, re, numpy as np
P = "pipe/"; out = []
def foldcos(f):
    if not os.path.exists(f): return None
    l = [x for x in open(f) if "fold-vs-fold subspace cosines" in x]
    return [float(v) for v in l[-1].split(":")[-1].split()] if l else None
for ds in ("J4001", "J207", "J2708", "J399"):
    base = "rae_base_control" if ds == "J4001" else "rae_base"
    for run in [base] + [f"start_{m}" for m in ("free", "vq", "mlp", "diffmap", "latent")]:
        d = P + f"{ds}/{run}/"
        if not os.path.exists(d + "DONE"): continue
        r = dict(ds=ds, run=run)
        r["null_fold"] = foldcos(d + "start/ppca_null.txt")
        cand = glob.glob(d + "ppca/*.txt") + [d + "ppca.log"]; r["cond_fold"] = next((foldcos(f) for f in cand if foldcos(f)), None)
        its = [json.loads(l[l.index("IT {") + 3:]) for l in open(d + "iter2.log") if "IT {" in l] if os.path.exists(d + "iter2.log") else []
        r["rounds"] = [(it["round"], round(it["fold_cosines"][0], 2), round(it["pct_of_main"], 1)) for it in its]
        if os.path.exists(d + "e2e_sharpen.json"):
            e = json.load(open(d + "e2e_sharpen.json")); r["r2_start_only"] = round(float(np.mean(e["r2_pc1_only"])), 3); r["r2_plus_resid"] = round(float(np.mean(e["r2_pc1_plus_residual"])), 3)
        if os.path.exists(d + "iter2/S_final.npy"):
            S = np.load(d + "iter2/S_final.npy"); C = np.corrcoef(S.T) if S.shape[1] > 1 else np.ones((1, 1)); off = np.abs(C - np.diag(np.diag(C)))
            r["n_coords"] = S.shape[1]; r["max_offdiag_corr"] = round(float(off.max()), 3) if S.shape[1] > 1 else None
            sc = np.load(d + "pc.scores.npy")[:, 0]; r["corr_S0_start"] = round(abs(float(np.corrcoef(S[:, 0], sc)[0, 1])), 3)
        out.append(r)
json.dump(out, open("od/qstats.json", "w"), indent=1)
for r in out:
    nf = r["null_fold"]; cf = r["cond_fold"]
    print(f"{r['ds']:6s} {r['run']:17s} null fold {('%.2f %.2f %.2f' % tuple(nf[:3])) if nf else '   -':15s} cond fold {('%.2f %.2f %.2f' % tuple(cf[:3])) if cf else '   -':15s}"
          f" r2 start {r.get('r2_start_only', '-')} +resid {r.get('r2_plus_resid', '-')} | coords {r.get('n_coords', '-')} maxcorr {r.get('max_offdiag_corr', '-')} S0~start {r.get('corr_S0_start', '-')}"
          f" | rounds(r, fold1, %main) {r['rounds'][:6]}")
