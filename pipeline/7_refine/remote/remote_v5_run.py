"""(11-25) east wrapper: builds the `cryosparcm cli` expression that execs remote_v5.py.  Usage: remote_v5_run.py create STATE TITLE STAR...  |  status STATE..."""
import json, os, subprocess, sys
CS = "/data1/sergey/PROGRAM/CRYOSPARC2/cryosparc2_master/bin/cryosparcm"; H = os.path.dirname(os.path.abspath(__file__)); cmd = sys.argv[1]
g = dict(CMD=cmd, REFP=H + "/refp.json")
if cmd == "create": g.update(STATE=sys.argv[2], TITLE=sys.argv[3], STARS=sys.argv[4:])
else: g.update(STATES=sys.argv[2:])
expr = "(lambda g: (exec(open(%r).read(), g), g['RESULT'])[1])(dict(api=api, **%r))" % (H + "/remote_v5.py", g)
r = subprocess.run([CS, "cli", expr], capture_output=True, text=True, env=dict(os.environ, CRYOSPARC_LOG_DIR=os.path.expanduser("~/cslog")))
line = (r.stdout.strip().split("\n") or [""])[-1]
try: line = json.loads(line)
except Exception: pass
print(line if isinstance(line, str) else json.dumps(line), flush=True)
if not str(line).startswith(("STATUS", "CREATED")): print(r.stdout[-3000:], r.stderr[-3000:], file=sys.stderr)
