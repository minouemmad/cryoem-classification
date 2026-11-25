import json, glob
S = '/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn/J4001_gP25_WT/candidate_groups_epoch50/axis_survey'; out = []
for f in glob.glob(S + '/cryosparc_jobs_kcP17_*.json'):
    for v in json.load(open(f)).values():
        try:
            if api.jobs.get_status(v['project'], v['nu']).value == 'completed': out.append(v['nu'])
        except Exception: pass
json.dump(out, open('/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn/J4001_gP25_WT/method_rank/p17_completed.json', 'w')); RESULT = f'{len(out)} completed'
