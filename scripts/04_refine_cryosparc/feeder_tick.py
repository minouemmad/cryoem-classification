# One feeder tick for ALL centre-scan / exploration chains in project P17 (run inside `cryosparcm cli` with...
# file names.
import glob, json, os
S = '/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn/J4001_gP25_WT/candidate_groups_epoch50/axis_survey'
CAP = {'ab': 2, 'nu': 1}; GPU = 2; ACTIVE = ('queued', 'launched', 'started', 'running', 'waiting')
order = []
for f in sorted(glob.glob(f'{S}/cryosparc_jobs_kcP17_*.json')):
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
active = {}
for p, j, kind, parent in order:
    if status(p, j) in ACTIVE:
        active[kind] = active.get(kind, 0) + 1
enq, blocked = [], []
_v = os.statvfs('/data'); FREE_GB = _v.f_bavail * _v.f_frsize / 1e9; LOWDISK = FREE_GB < 300
for p, j, kind, parent in order:
    if status(p, j) != 'building':
        continue
    ps = status(p, parent)
    if ps in ('failed', 'killed'):
        blocked.append(j); continue
    if LOWDISK or ps != 'completed' or active.get(kind, 0) >= CAP[kind]:
        continue
    api.jobs.enqueue(p, j, hostname='hudson', gpus=[GPU])
    active[kind] = active.get(kind, 0) + 1; st[(p, j)] = 'queued'; enq.append(j)
from collections import Counter
c = Counter(status(p, j) for p, j, *_ in order)
RESULT = f"free {FREE_GB:.0f}GB{' LOWDISK' if LOWDISK else ''} enqueued {enq} blocked {blocked} | {dict(c)} | active {active}"
