# Run inside `cryosparcm cli` with globals api, PROJECT, STARS (leaf STAR paths), STATE (json path), TITLE.
# Per recon-tree leaf: import particles -> ab initio, 1 class (params of P7 J637) -> NU refinement (P7 J647), created in a
# new workspace (building, not queued). Job uids -> STATE; leaf_chain.sh queues them.
import json, os
if os.path.exists(STATE):
    RESULT = 'exists: ' + open(STATE).read()
else:
    skip = ('random_seed', 'random_seed_default')
    ref = lambda j: {k: v for k, v in api.jobs.find_one('P7', j).model_dump()['spec']['params'].items() if v is not None and k not in skip}
    P_ab, P_nu = ref('J637'), ref('J647')
    W = api.workspaces.create(PROJECT, title=TITLE).uid
    st = dict(project=PROJECT, workspace=W, params_from=dict(abinit='P7 J637', nu='P7 J647'), leaves={})
    for s in STARS:
        nm = os.path.basename(s)[:-5]
        imp = api.jobs.create(PROJECT, W, {'particle_meta_path': s, 'particle_blob_path': '/', 'sign': -1, 'enable_validation': True,
                                           'blob_exists': True, 'ctf_exists': True, 'location_exists': False,
                                           'alignments3D_exists': False}, type='import_particles', title=f'{TITLE}: {nm}')
        ab = api.jobs.create(PROJECT, W, P_ab, type='homo_abinit', title=f'{TITLE}: {nm} ab initio')
        api.jobs.connect(PROJECT, ab.uid, 'particles', source_output_name='imported_particles', source_job_uid=imp.uid)
        nu = api.jobs.create(PROJECT, W, P_nu, type='nonuniform_refine_new', title=f'{TITLE}: {nm} NU')
        api.jobs.connect(PROJECT, nu.uid, 'particles', source_output_name='particles_class_0', source_job_uid=ab.uid)
        api.jobs.connect(PROJECT, nu.uid, 'volume', source_output_name='volume_class_0', source_job_uid=ab.uid)
        st['leaves'][nm] = dict(star=s, import_particles=imp.uid, abinit=ab.uid, nu=nu.uid)
    json.dump(st, open(STATE, 'w'), indent=1)
    RESULT = json.dumps(st)
