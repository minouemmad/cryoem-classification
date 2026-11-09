"""Datasets processed by the dataset pipeline: source heteroref exports in data/, cryoSPARC project, particle
stacks and starting map. J4001 (gP25 WT) and J399 (hP7 E1371Q ATP) were set up before this pipeline.
"""
D = "/data/hCFTR/mae2183/cryoem-classification/data"
R = "/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn"
V1 = "/mnt/nfs_HuntLabSynologyDS3622CryoEM_V1/hCFTR/cryoSPARC"

SETS = {
    "J4624_eP37_WT_T2a": dict(
        project="P5",   # hCFTR_WT+T2a_from_eP37 (the J4624 heteroref is an imported job there)
        project_dir=f"{V1}/CS-hcftr-wt-t2a-from-ep37",
        multi=f"{D}/J4624_eP37_WT_T2a/heteroref_J4624/cryosparc_P37_J4624_00062_particles_alignments3D_multi.cs",
        passthrough=[f"{D}/J4624_eP37_WT_T2a/heteroref_J4624/cryosparc_P37_J4624_passthrough_particles_all_classes_{k}.cs"
                     for k in ("blob", "ctf", "alignments3D", "location")],
        stack_root=f"{V1}/CS-hcftr-wt-t2a-from-ep37/imports/jobs/J4624_hetero_refine/J4624_particles_all_classes",
        init_volume=f"{D}/J4624_eP37_WT_T2a/heteroref_J4624/heteroref_volumes/cryosparc_P37_J4624_class_08_00062_volume_sharp.mrc",
        heteroref="J4624"),
    "J2708_eP30_F508del_E1371W": dict(
        project="P6",   # hCFTR_F508del-E1371W-6SS_from_eP30 (J2760 split_0 = the 231,930 J2708 particles)
        project_dir=f"{V1}/CS-hcftr-f508del-from-ep30/CS-hcftr-f508del-e1371w-6ss-from-ep30",
        multi=f"{D}/J2708_eP30_F508del_E1371W/heteroref_J2708/cryosparc_P30_J2708_00042_particles_alignments3D_multi.cs",
        passthrough=[f"{D}/J2708_eP30_F508del_E1371W/heteroref_J2708/cryosparc_P30_J2708_passthrough_particles_all_classes_{k}.cs"
                     for k in ("blob", "ctf", "alignments3D", "location")],
        stack_root=f"{V1}/CS-hcftr-f508del-from-ep30/CS-hcftr-f508del-e1371w-6ss-from-ep30/imports/jobs/J2760_particle_sets/J2760_split_0",
        init_volume=f"{D}/J2708_eP30_F508del_E1371W/3dva_J2025/cryosparc_P30_J2025_map.mrc",
        heteroref="J2708"),
    "J207_hP12_IDOR4": dict(
        project="P12",  # hCFTR_m26may26f_IDOR4 (native project: J76 extract)
        project_dir="/mnt/nfs_HuntLabSynologyDS3622CryoEM_V2/hCFTR/cS/CS-hcftr-m26may26f-idor4",
        multi=f"{D}/J207_hP12_IDOR4/heteroref_J207/cryosparc_P12_J207_particles_alignments3D_multi.cs",
        passthrough=[f"{D}/J207_hP12_IDOR4/heteroref_J207/cryosparc_P12_J207_passthrough_particles_all_classes_{k}.cs"
                     for k in ("blob", "ctf", "location")],
        stack_root="/mnt/nfs_HuntLabSynologyDS3622CryoEM_V2/hCFTR/cS/CS-hcftr-m26may26f-idor4",
        init_volume="/mnt/nfs_HuntLabSynologyDS3622CryoEM_V2/hCFTR/cS/CS-hcftr-m26may26f-idor4/J106/J106_008_volume_map.mrc",
        heteroref="J207"),
}
FIRST_PROTEIN_CLASS = 6   # heteroref classes 0-5 are the decoy/junk references (<2 % of particles in eP37/eP30), as in J1442 -> J4001
SEED = 44688              # cryoDRGN seed of the J4001 train_fullset_128 runs
