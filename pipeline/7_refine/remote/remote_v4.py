"""(11-25) label-free class chains (import -> ab initio -> NU) on a cryoSPARC v4.x instance (ganges) through `cryosparcm cli` (keyword calls, ~4 s each).
Usage: remote_v4.py create STATE TITLE STAR [STAR ...]  |  remote_v4.py status STATE [STATE ...]  (prints one JSON line)
Params = hudson P17 J451 / J452 (refp.json), SSD cache off (ganges /scr NVMe is dead), minus parameters this version lacks; import without `sign` (v4 rejects -1)."""
import json, os, subprocess, sys
CS = "/data/emsoft/cryosparc/cryosparc_master/bin/cryosparcm"; UID = "6a1b333604003f5fc2988042"; CONT = "/data/mae2183_cs/od_cs"
H = os.path.dirname(os.path.abspath(__file__)); REF = json.load(open(H + "/refp.json"))
SKIP = {"abinit_window_mode", "do_dynamic_mask", "refine_dynamic_mask_near_multiplier", "refine_dynamic_mask_far_multiplier", "refine_ews_zsign", "refine_ews_simple"}
def cli(e):
    r = subprocess.run([CS, "cli", e], capture_output=True, text=True); return (r.stdout + r.stderr).strip().split("\n")[-1].strip()
def project():
    f = H + "/project.json"
    if os.path.exists(f): return json.load(open(f))
    os.makedirs(CONT, exist_ok=True)
    p = cli(f"create_empty_project(owner_user_id={UID!r}, project_container_dir={CONT!r}, title='label-free classes od')"); assert p.startswith("P"), p
    w = cli(f"create_empty_workspace(project_uid={p!r}, created_by_user_id={UID!r}, title='label-free classes')"); assert w.startswith("W"), w
    d = dict(project=p, workspace=w, dir=cli(f"get_project_dir_abs(project_uid={p!r})")); json.dump(d, open(f, "w")); return d
cmd = sys.argv[1]
if cmd == "create":
    STATE, TITLE, STARS = sys.argv[2], sys.argv[3], sys.argv[4:]; pr = project(); P, W = pr["project"], pr["workspace"]
    st = json.load(open(STATE)) if os.path.exists(STATE) else {}
    pa = {k: v for k, v in dict(REF["J451"], compute_use_ssd=False).items() if k not in SKIP}; pn = {k: v for k, v in dict(REF["J452"], compute_use_ssd=False).items() if k not in SKIP}
    for s in STARS:
        nm = os.path.basename(s)[:-5]
        if nm in st: continue
        t = f"{TITLE} {nm}"
        imp = cli(f"make_job(job_type='import_particles', project_uid={P!r}, workspace_uid={W!r}, user_id={UID!r}, title={t!r}, params=dict(particle_meta_path={s!r}, particle_blob_path='/', enable_validation=True, blob_exists=True, ctf_exists=True, location_exists=False, alignments3D_exists=False))"); assert imp.startswith("J"), imp
        ab = cli(f"make_job(job_type='homo_abinit', project_uid={P!r}, workspace_uid={W!r}, user_id={UID!r}, title={t!r}, params={pa!r}, input_group_connects=dict(particles='{imp}.imported_particles'))"); assert ab.startswith("J"), ab
        nu = cli(f"make_job(job_type='nonuniform_refine_new', project_uid={P!r}, workspace_uid={W!r}, user_id={UID!r}, title={t!r}, params={pn!r}, input_group_connects=dict(particles='{ab}.particles_class_0', volume='{ab}.volume_class_0'))"); assert nu.startswith("J"), nu
        for j in (imp, ab, nu): cli(f"enqueue_job(project_uid={P!r}, job_uid={j!r}, lane='default', user_id={UID!r})")
        st[nm] = dict(project=P, dir=pr["dir"], import_job=imp, abinit=ab, nu=nu, star=s, status=["queued"] * 3); json.dump(st, open(STATE, "w"), indent=1); print("chain", nm, imp, ab, nu, flush=True)
    print("CREATED", len(st), flush=True)
elif cmd == "status":
    out = {}
    for STATE in sys.argv[2:]:
        if not os.path.exists(STATE): continue
        st = json.load(open(STATE)); ch = False
        for nm, v in st.items():
            if v["status"][2] not in ("completed", "failed", "killed") and v["status"][0] != "failed" and v["status"][1] != "failed":
                v["status"] = [cli(f"get_job_status(project_uid={v['project']!r}, job_uid={v[k]!r})") for k in ("import_job", "abinit", "nu")]; ch = True
            out.setdefault(STATE, {})[nm] = dict(nu=v["nu"], dir=v["dir"], status=v["status"])
        if ch: json.dump(st, open(STATE, "w"), indent=1)
    print("STATUS " + json.dumps(out))
