# J4624 group chains in project PROJ / workspace WS: import (STAR without poses, dataset/write_star.py) ->...
# Ab initio / NU created but NOT enqueued (feeder_tick.py).
import json, os
st = json.load(open(STATE)) if os.path.exists(STATE) else {}
skip = ('random_seed', 'random_seed_default')
ref = lambda p, j: {k: v for k, v in api.jobs.find_one(p, j).model_dump()['spec']['params'].items() if v is not None and k not in skip}
P_ab, P_nu = ref(REF_PROJ, REF_AB), ref(REF_PROJ, REF_NU); new = 0
for s in STARS:
    nm = os.path.basename(s)[:-5]
    if nm in st:
        continue
    title = 'J4624 residual-split test ' + nm
    imp = api.jobs.create(PROJ, WS, {'particle_meta_path': s, 'particle_blob_path': '/', 'sign': -1, 'enable_validation': True, 'blob_exists': True, 'ctf_exists': True, 'location_exists': False, 'alignments3D_exists': False},
                          type='import_particles', title=title)
    ab = api.jobs.create(PROJ, WS, P_ab, type='homo_abinit', title=title)
    api.jobs.connect(PROJ, ab.uid, 'particles', source_output_name='imported_particles', source_job_uid=imp.uid)
    nu = api.jobs.create(PROJ, WS, P_nu, type='nonuniform_refine_new', title=title)
    api.jobs.connect(PROJ, nu.uid, 'particles', source_output_name='particles_class_0', source_job_uid=ab.uid)
    api.jobs.connect(PROJ, nu.uid, 'volume', source_output_name='volume_class_0', source_job_uid=ab.uid)
    api.jobs.enqueue(PROJ, imp.uid, hostname='hudson')
    st[nm] = dict(project=PROJ, workspace=WS, import_job=imp.uid, abinit=ab.uid, nu=nu.uid, gpu=GPUS[0], name=nm, star=s, dataset='J4624')
    json.dump(st, open(STATE, 'w'), indent=1); new += 1
RESULT = f'{new} new chains; {len(st)} total in {STATE}'
