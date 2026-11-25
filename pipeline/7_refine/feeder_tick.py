# One feeder tick (inside `cryosparcm cli`, globals api, STATES, HOST, GPU): over all chains in the STATES glob (sorted file names = priority),
# enqueue ab initio once its import completed (cap 2 running) and NU once its ab initio completed (cap 2); pause below 300 GB free on /data.
import glob, json, os
from collections import Counter
CAP = {'ab': 2, 'nu': 2}; ACTIVE = ('queued', 'launched', 'started', 'running', 'waiting'); order = []
for f in sorted(glob.glob(STATES)):
    for v in json.load(open(f)).values():
        order += [(v['project'], v['abinit'], 'ab', v['import_job']), (v['project'], v['nu'], 'nu', v['abinit'])]
st = {}
def status(p, j):
    if (p, j) not in st:
        try:
            st[(p, j)] = api.jobs.get_status(p, j).value
        except Exception:
            st[(p, j)] = 'deleted'
    return st[(p, j)]
active = Counter(kind for p, j, kind, _ in order if status(p, j) in ACTIVE); v = os.statvfs('/data'); free = v.f_bavail * v.f_frsize / 1e9; enq, blocked = [], []
for p, j, kind, parent in order:
    if status(p, j) != 'building':
        continue
    if status(p, parent) in ('failed', 'killed'):
        blocked.append(j); continue
    if free < 300 or status(p, parent) != 'completed' or active[kind] >= CAP[kind]:
        continue
    api.jobs.enqueue(p, j, hostname=HOST, gpus=[GPU]); active[kind] += 1; st[(p, j)] = 'queued'; enq.append(j)
RESULT = f"free {free:.0f}GB enqueued {enq} blocked {blocked} | {dict(Counter(status(p, j) for p, j, *_ in order))} | active {dict(active)}"
