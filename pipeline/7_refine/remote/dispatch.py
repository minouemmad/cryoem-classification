#!/usr/bin/env python3
"""(11-25) Argon dispatcher for the label-free class sets of the non-cryoDRGN starts (start_free / vq / mlp / diffmap) + start_latent ( every K: J4001 J207 J2708 2-5, J399 2-10)
-> import + ab initio + NU on hudson (P17 + feeder_od), ganges (cryoSPARC v4, own project P34) or east (v5, P41), whichever has room.
Each pass: (1) hudson: finished pipeline runs, existing P17 set files, feeder backlog; (2) remote chain status; (3) finished remote NU maps -> hudson od/remote/<host>_<P>/<J>/
and axis_survey/cryosparc_jobs_remote_<host>_od_<DS>_<RUN>_K<K>.json (pdir) for od_compare; (4) pending sets (lowest K first) -> free machines.
Usage: dispatch.py [once]      state argon_dispatch/state.json, log argon_dispatch/dispatch.log  (kept in the project dir: login-node scratch is per node)"""
import json, os, re, shlex, subprocess, sys, time
D = os.path.dirname(os.path.abspath(__file__)); LOG = D + "/dispatch.log"; STF = D + "/state.json"
R = "/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn/"; B = R + "model_pipeline/"; A = R + "J4001_gP25_WT/candidate_groups_epoch50/axis_survey/"
PY = "/home/mae2183/miniconda3/envs/cryodrgn/bin/python"; HU = "mae2183@hudson.biology.columbia.edu"
KS = {"J4001": range(2, 6), "J207": range(2, 6), "J2708": range(2, 6), "J399": range(2, 11)}; DSO = list(KS); METH = ["free", "vq", "mlp", "diffmap", "latent"]   # (11-25) latent (cryoDRGN encoder PC1) added on request
SRC = {"J4001": R + "J4001_gP25_WT", "J207": R + "J207_hP12_IDOR4", "J2708": R + "J2708_eP30_F508del_E1371W", "J399": R + "J399_hP7_E1371Q_ATP"}
REM = {"ganges": dict(ssh="mae2183@ganges.biology.columbia.edu", data="/data/mae2183_cs/mp/data", stars="/data/mae2183_cs/od_stars", cap=6, run="python3 ~/remote_cs/od/remote_v4.py"),
       "east": dict(ssh="mae2183@east.biology.columbia.edu", data="/scratch/mae2183/mp/data", stars="/scratch/mae2183/od_stars", cap=6, run="python3 ~/remote_cs/od/remote_v5_run.py")}
EXF = D + "/extra_sets.txt"   # (11-25) pre-built sets (STARs + set.json already in od/<DS>/<RUN>_K<K>/, e.g. axis windows): "DS RUN K" per line, remote only, assigned first
def extra(): return [(a, b, int(c)) for a, b, c in (l.split() for l in open(EXF) if l.strip() and not l.startswith("#"))] if os.path.exists(EXF) else []
HCAP = int(os.environ.get("HCAP", "30")); FINAL = ("completed", "failed", "killed")
SSH = ["ssh", "-o", "ControlPath=" + os.path.expanduser("~/.ssh/cm-%r@%h"), "-o", "BatchMode=yes"]
def log(*a):
    s = time.strftime("%m-%d %H:%M ") + " ".join(str(x) for x in a); print(s, flush=True); open(LOG, "a").write(s + "\n")
def ssh(host, cmd, t=900):
    try: return subprocess.run(SSH + [host, cmd], capture_output=True, text=True, timeout=t).stdout
    except Exception as e: log("ssh error", host, repr(e)[:200]); return ""
def pipe(h1, c1, h2, c2, t=1800):
    q = lambda h, c: " ".join(shlex.quote(x) for x in SSH + [h, c])
    return subprocess.run(q(h1, c1) + " | " + q(h2, c2), shell=True, capture_output=True, text=True, timeout=t)
st = json.load(open(STF)) if os.path.exists(STF) else {"sets": {}, "failed": []}
def save(): json.dump(st, open(STF + ".tmp", "w"), indent=1); os.replace(STF + ".tmp", STF)
SNAP = r'''
import glob, json, os, re, subprocess
B, A = %r, %r; o = dict(done=sorted(d.split("/")[-3] + "/" + d.split("/")[-2] for d in glob.glob(B + "pipe/*/start_*/DONE")),
     p17=sorted(os.path.basename(f) for f in glob.glob(A + "cryosparc_jobs_kcP17_a_od_*.json")), feeder=subprocess.run(["pgrep", "-f", "feeder_od.sh"], capture_output=True).returncode == 0)
l = open(A + "feeder_od.log").read().strip().split("\n")[-1]; m = re.search(r"'building': (\d+)", l); b = re.search(r"blocked \[([^\]]*)\]", l)
o.update(building=int(m.group(1)) if m else 0, blocked=len(re.findall(r"J\d+", b.group(1))) if b else 0, last=l[:16]); print("SNAP" + json.dumps(o))
''' % (B, A)
def snap():
    out = ssh(HU, f"{PY} -c {shlex.quote(SNAP)}")
    for l in out.split("\n"):
        if l.startswith("SNAP"): return json.loads(l[4:])
    log("hudson snapshot failed", out[-300:]); return None
def rstatus(h):
    sets = [k for k, v in st["sets"].items() if v["host"] == h]
    if not sets: return
    out = ssh(REM[h]["ssh"], REM[h]["run"] + " status " + " ".join(st["sets"][k]["dir"] + "/state.json" for k in sets), t=1500)
    l = [x for x in out.split("\n") if x.startswith("STATUS ")]
    if not l: log(h, "status failed", out[-300:]); return
    res = json.loads(l[-1][7:])
    for k in sets:
        v = st["sets"][k]; v["chains"] = res.get(v["dir"] + "/state.json", v.get("chains", {}))
def inflight(h): return sum(v["K"] - sum(c["status"][2] in FINAL or "failed" in c["status"][:2] for c in v.get("chains", {}).values()) for v in st["sets"].values() if v["host"] == h)
def ready(h):
    out = ssh(REM[h]["ssh"], f"ls {REM[h]['data']}/*/particles.128.mrcs"); return {p.split("/")[-2] for p in out.split()}
def fetch(h):
    for k, v in st["sets"].items():
        if v["host"] != h: continue
        ds, run, K = k.split("|"); ch = v.get("chains", {}); new = False
        for nm, c in ch.items():
            if "failed" in c["status"] or "killed" in c["status"]:
                if nm not in st["failed"]: st["failed"].append(nm); log("FAILED", h, nm, c["status"])
                continue
            if c["status"][2] != "completed" or nm in v["fetched"]: continue
            pd = B + f"od/remote/{h}_{os.path.basename(c['dir'])}"   # c["dir"] = remote project dir
            r = pipe(REM[h]["ssh"], f"cd {c['dir']} && tar -cf - {c['nu']}/*.mrc", HU, f"mkdir -p {pd} && tar -C {pd} -xf - && ls {pd}/{c['nu']}/*_volume_map_half_B.mrc")
            if "half_B" in r.stdout: v["fetched"].append(nm); new = True; log("fetched", h, nm, c["nu"])
            else: log("fetch failed", h, nm, r.stderr[-200:])
        if new or (ch and not v.get("hfile")):
            pd = B + f"od/remote/{h}_{os.path.basename(next(iter(ch.values()))['dir'])}"
            js = json.dumps({nm: dict(nu=c["nu"], pdir=pd, host=h) for nm, c in ch.items()}, indent=1)
            subprocess.run(SSH + [HU, f"cat > {A}cryosparc_jobs_remote_{h}_od_{ds}_{run}_K{K}.json"], input=js, text=True, timeout=300); v["hfile"] = True
def assign_remote(h, ds, run, K, pre=False):
    out = ssh(HU, ("" if pre else f"cd {B}od && {PY} od_stars.py {ds} {run} {K} > /dev/null 2>&1; ") + f"ls {B}od/{ds}/{run}_K{K}/cryosparc/*.star | wc -l")
    if out.strip() != str(K): log("stars failed", ds, run, K, out[-200:]); return False
    X = f"{REM[h]['stars']}/{ds}_{run}_K{K}"; old = SRC[ds] + "/inputs/particles.128.mrcs"; new = f"{REM[h]['data']}/{ds}/particles.128.mrcs"
    r = pipe(HU, f"cd {B}od/{ds}/{run}_K{K}/cryosparc && tar -cf - *.star", REM[h]["ssh"],
             f"mkdir -p {X} && tar -C {X} -xf - && sed -i 's#{old}#{new}#' {X}/*.star && echo OLD $(grep -c '{old}' {X}/*.star | awk -F: '{{s+=$2}} END{{print s}}') NEW $(grep -h -c '{new}' {X}/*.star | awk '{{s+=$1}} END{{print s}}')")
    m = re.search(r"OLD (\d+) NEW (\d+)", r.stdout)
    if not m or m.group(1) != "0" or m.group(2) == "0": log("star rewrite failed", h, ds, run, K, r.stdout[-200:], r.stderr[-200:]); return False
    ssh(REM[h]["ssh"], f"cd {X} && (setsid nohup {REM[h]['run']} create {X}/state.json '{ds} label-free classes' {X}/*.star > {X}/create.log 2>&1 < /dev/null &)")
    st["sets"][f"{ds}|{run}|{K}"] = dict(host=h, K=K, dir=X, t=time.time(), fetched=[]); save(); log("assigned", h, ds, run, f"K{K}", m.group(2), "particles"); return True
def assign_hudson(ds, run, K):
    ssh(HU, f"bash {B}od/od_submit.sh {ds} {run} {K}"); st["sets"][f"{ds}|{run}|{K}"] = dict(host="hudson", K=K, t=time.time()); save(); log("assigned hudson", ds, run, f"K{K}")
while True:
    s = snap()
    if s:
        if not s["feeder"] and s["building"] > s["blocked"]:
            ssh(HU, f"setsid nohup bash {A}var_windows/feeder_od.sh > /dev/null 2>&1 < /dev/null &"); log("restarted hudson feeder")
        for h in REM: rstatus(h); fetch(h)
        save()
        done = set(s["done"]); runs = [(ds, f"start_{m}") for ds in DSO for m in METH]
        pend = sorted([(K, DSO.index(ds), METH.index(run[6:]), ds, run) for ds, run in runs if f"{ds}/{run}" in done for K in KS[ds]
                       if f"{ds}|{run}|{K}" not in st["sets"] and f"cryosparc_jobs_kcP17_a_od_{ds}_{run}_K{K}.json" not in s["p17"]])
        backlog = s["building"] - s["blocked"]; rdy = {h: ready(h) for h in REM}; fl = {h: inflight(h) for h in REM}; nassigned = 0
        ext = [x for x in extra() if "|".join(map(str, x)) not in st["sets"]]
        for ds, run, K in list(ext):
            hs = sorted((fl[h] - REM[h]["cap"], h) for h in REM if ds in rdy[h] and fl[h] < REM[h]["cap"])
            if hs and assign_remote(hs[0][1], ds, run, K, pre=True): fl[hs[0][1]] += K; ext.remove((ds, run, K))
        for K, _, _, ds, run in pend:
            if backlog + 2 * K <= HCAP: assign_hudson(ds, run, K); backlog += 2 * K; nassigned += 1; continue
            hs = sorted((fl[h] - REM[h]["cap"], h) for h in REM if ds in rdy[h] and fl[h] < REM[h]["cap"])
            if hs and assign_remote(hs[0][1], ds, run, K): fl[hs[0][1]] += K; nassigned += 1
        log(f"pass: runs done {sum(f'{d}/{r}' in done for d, r in runs)}/{len(runs)} | pending sets {len(pend) - nassigned} | hudson backlog {backlog} feeder {'on' if s['feeder'] else 'off'} | "
            + " ".join(f"{h} inflight {inflight(h)} fetched {sum(len(v.get('fetched', [])) for v in st['sets'].values() if v['host'] == h)}" for h in REM) + f" | extra pending {len(ext)} | failed {len(st['failed'])}")
        alldone = all(f"{d}/{r}" in done for d, r in runs) and len(pend) == nassigned and not ext and all(inflight(h) == 0 for h in REM) and not s["feeder"] \
                  and all(len(v["fetched"]) + sum(n in st["failed"] for n in v.get("chains", {})) >= v["K"] for v in st["sets"].values() if v["host"] != "hudson")
        if alldone: ssh(HU, f"touch {B}od/cmp/dispatch.done"); log("ALL DONE"); break
    if len(sys.argv) > 1 and sys.argv[1] == "once": break
    time.sleep(300)
