# Run inside `cryosparcm cli` (see submit.sh). Per STAR: import particles (no poses) -> ab initio (1 class) -> NU refinement, parameters copied
# from reference jobs REF_AB / REF_NU of REF_PROJ.  Imports are queued; ab initio / NU stay 'building' until feeder_tick.py enqueues them.
# Globals: api, PROJ, WS, HOST, GPU, REF_PROJ, REF_AB, REF_NU, STARS, STATE, TITLE, DATASET.   STATE (json) maps STAR name -> job uids.
import json, os
st = json.load(open(STATE)) if os.path.exists(STATE) else {}
ref = lambda p, j: {k: v for k, v in api.jobs.find_one(p, j).model_dump()['spec']['params'].items() if v is not None and k not in ('random_seed', 'random_seed_default')}
P_ab, P_nu = ref(REF_PROJ, REF_AB), ref(REF_PROJ, REF_NU); new = 0
for s in STARS:
    nm = os.path.basename(s)[:-5]
    if nm in st:
        continue
    t = TITLE + ' ' + nm
    imp = api.jobs.create(PROJ, WS, {'particle_meta_path': s, 'particle_blob_path': '/', 'sign': -1, 'enable_validation': True, 'blob_exists': True, 'ctf_exists': True,
                                     'location_exists': False, 'alignments3D_exists': False}, type='import_particles', title=t)
    ab = api.jobs.create(PROJ, WS, P_ab, type='homo_abinit', title=t)
    api.jobs.connect(PROJ, ab.uid, 'particles', source_output_name='imported_particles', source_job_uid=imp.uid)
    nu = api.jobs.create(PROJ, WS, P_nu, type='nonuniform_refine_new', title=t)
    api.jobs.connect(PROJ, nu.uid, 'particles', source_output_name='particles_class_0', source_job_uid=ab.uid)
    api.jobs.connect(PROJ, nu.uid, 'volume', source_output_name='volume_class_0', source_job_uid=ab.uid)
    api.jobs.enqueue(PROJ, imp.uid, hostname=HOST)
    st[nm] = dict(project=PROJ, workspace=WS, import_job=imp.uid, abinit=ab.uid, nu=nu.uid, gpu=GPU, name=nm, star=s, dataset=DATASET)
    json.dump(st, open(STATE, 'w'), indent=1); new += 1
RESULT = f'{new} new chains; {len(st)} total in {STATE}'
