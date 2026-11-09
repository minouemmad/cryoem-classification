# Run inside `cryosparcm cli` with globals api, NAME, PROJECT. Creates (building, not queued) in a new workspace:
# import particles + starting map -> homogeneous refinement (params of P7 J399) -> backprojection (P7 J825) -> NU refinement
# (P7 J647). Job uids -> results_cryodrgn/NAME/nu_consensus/cryosparc_job.json; run_pipeline.sh queues them.
import json, os
R = '/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn'
N = f'{R}/{NAME}/nu_consensus'
state = f'{N}/cryosparc_job.json'
if os.path.exists(state):
    RESULT = 'exists: ' + open(state).read()
else:
    skip = ('random_seed', 'random_seed_default')
    ref = lambda j: {k: v for k, v in api.jobs.find_one('P7', j).model_dump()['spec']['params'].items() if v is not None and k not in skip}
    P_homo, P_bp, P_nu = ref('J399'), ref('J825'), ref('J647')
    prep = json.load(open(f'{N}/prep.json'))
    ws = api.workspaces.create(PROJECT, title='cryoDRGN pose prep: homo refine -> backprojection -> NU (protein classes)')
    W = ws.uid
    t = f'{NAME} protein classes'
    imp = api.jobs.create(PROJECT, W, {'particle_meta_path': f'{N}/{NAME}_particles.cs', 'enable_validation': True,
                                       'blob_exists': True, 'ctf_exists': True, 'location_exists': True,
                                       'alignments3D_exists': False}, type='import_particles', title=f'{t} ({prep["n_protein"]})')
    vol = api.jobs.create(PROJECT, W, {'volume_blob_path': f'{N}/init_volume.mrc', 'volume_out_name': 'map'}, type='import_volumes', title=f'{t}: starting map')
    homo = api.jobs.create(PROJECT, W, P_homo, type='homo_refine_new', title=f'{t}: homogeneous refinement')
    api.jobs.connect(PROJECT, homo.uid, 'particles', source_output_name='imported_particles', source_job_uid=imp.uid)
    api.jobs.connect(PROJECT, homo.uid, 'volume', source_output_name='imported_volume_1', source_job_uid=vol.uid)
    bp = api.jobs.create(PROJECT, W, P_bp, type='homo_reconstruct', title=f'{t}: backprojection of homo-refine poses')
    api.jobs.connect(PROJECT, bp.uid, 'particles', source_output_name='particles', source_job_uid=homo.uid)
    nu = api.jobs.create(PROJECT, W, P_nu, type='nonuniform_refine_new', title=f'{t}: NU consensus (poses for cryoDRGN)')
    api.jobs.connect(PROJECT, nu.uid, 'particles', source_output_name='particles', source_job_uid=homo.uid)
    api.jobs.connect(PROJECT, nu.uid, 'volume', source_output_name='volume', source_job_uid=bp.uid)
    d = dict(project=PROJECT, workspace=W, import_particles=imp.uid, import_volume=vol.uid, homo=homo.uid, backproject=bp.uid,
             nu=nu.uid, params_from=dict(homo='P7 J399', backproject='P7 J825', nu='P7 J647'))
    json.dump(d, open(state, 'w'), indent=1)
    RESULT = json.dumps(d)
