# (11-25) label-free class chains on a cryoSPARC v5 instance; exec'd inside `cryosparcm cli` (globals: api, CMD, STATES / STATE, TITLE, STARS, REFP).
# Params = hudson P17 J451 / J452 (refp.json), SSD cache off; any parameter this version rejects is skipped (recorded per chain).
import json, os
P, W, UID = "P41", "W1", "6a1b3216b07a5889f9e0debf"
def mk(params, **kw):
    try: return api.jobs.create(P, W, params, **kw), []
    except Exception:
        j = api.jobs.create(P, W, {}, **kw); bad = []
        for k, v in params.items():
            try: api.jobs.set_param(P, j.uid, k, v)
            except Exception: bad.append(k)
        return j, bad
if CMD == "create":
    REF = json.load(open(REFP)); st = json.load(open(STATE)) if os.path.exists(STATE) else {}; pdir = api.projects.find_one(P).project_dir
    pa = dict(REF["J451"], compute_use_ssd=False); pn = dict(REF["J452"], compute_use_ssd=False); new = 0
    for s in STARS:
        nm = os.path.basename(s)[:-5]
        if nm in st: continue
        t = f"{TITLE} {nm}"
        imp, _ = mk({"particle_meta_path": s, "particle_blob_path": "/", "sign": -1, "enable_validation": True, "blob_exists": True, "ctf_exists": True, "location_exists": False, "alignments3D_exists": False}, type="import_particles", title=t)
        ab, b1 = mk(pa, type="homo_abinit", title=t); api.jobs.connect(P, ab.uid, "particles", source_output_name="imported_particles", source_job_uid=imp.uid)
        nu, b2 = mk(pn, type="nonuniform_refine_new", title=t)
        api.jobs.connect(P, nu.uid, "particles", source_output_name="particles_class_0", source_job_uid=ab.uid); api.jobs.connect(P, nu.uid, "volume", source_output_name="volume_class_0", source_job_uid=ab.uid)
        for j in (imp, ab, nu): api.jobs.enqueue(P, j.uid, lane="default")
        st[nm] = dict(project=P, dir=pdir, import_job=imp.uid, abinit=ab.uid, nu=nu.uid, star=s, skipped=b1 + b2, status=["queued"] * 3); json.dump(st, open(STATE, "w"), indent=1); new += 1
    RESULT = f"CREATED {new} of {len(st)}"
else:
    out = {}
    for STATE in STATES:
        if not os.path.exists(STATE): continue
        st = json.load(open(STATE))
        for nm, v in st.items():
            if v["status"][2] not in ("completed", "failed", "killed") and "failed" not in v["status"][:2]:
                v["status"] = [api.jobs.find_one(P, v[k]).model_dump(mode="json")["status"] for k in ("import_job", "abinit", "nu")]
            out.setdefault(STATE, {})[nm] = dict(nu=v["nu"], dir=v["dir"], status=v["status"])
        json.dump(st, open(STATE, "w"), indent=1)
    RESULT = "STATUS " + json.dumps(out)
