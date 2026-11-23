# import -> ab initio -> NU for every not-yet-submitted entry of MANIFEST, created in project PROJ /...
# REF_AB / REF_NU. Ab initio / NU are created but NOT enqueued: feeder_tick.py enqueues them a few at a time.
import json, os
D = '/mnt/nfs_HuntLabSynologyDS1621CryoEM/hCFTR/Movies/24sep20b/24sep20b/rawdata/CS-24sep20b-wt-control-white-from-january'
man = json.load(open(MANIFEST))
st = json.load(open(STATE)) if os.path.exists(STATE) else {}
skip = ('random_seed', 'random_seed_default')
ref = lambda p, j: {k: v for k, v in api.jobs.find_one(p, j).model_dump()['spec']['params'].items() if v is not None and k not in skip}
P_ab, P_nu = ref(REF_PROJ, REF_AB), ref(REF_PROJ, REF_NU)
new = 0
for m in man:
    nm = m['name']
    if nm in st:
        continue
    title = 'axis survey ' + nm
    imp = api.jobs.create(PROJ, WS, {'particle_meta_path': f"{D}/gmm_exports/recon_tree_v3/{nm}.star", 'sign': -1, 'enable_validation': True, 'blob_exists': True, 'ctf_exists': True,
                                     'alignments3D_exists': True}, type='import_particles', title=title)
    ab = api.jobs.create(PROJ, WS, P_ab, type='homo_abinit', title=title)
    api.jobs.connect(PROJ, ab.uid, 'particles', source_output_name='imported_particles', source_job_uid=imp.uid)
    nu = api.jobs.create(PROJ, WS, P_nu, type='nonuniform_refine_new', title=title)
    api.jobs.connect(PROJ, nu.uid, 'particles', source_output_name='particles_class_0', source_job_uid=ab.uid)
    api.jobs.connect(PROJ, nu.uid, 'volume', source_output_name='volume_class_0', source_job_uid=ab.uid)
    api.jobs.enqueue(PROJ, imp.uid, hostname='hudson')
    st[nm] = dict(project=PROJ, workspace=WS, import_job=imp.uid, abinit=ab.uid, nu=nu.uid, gpu=GPUS[0], **m)
    json.dump(st, open(STATE, 'w'), indent=1)
    new += 1
RESULT = f'{new} new chains; {len(st)} total in {STATE}'
