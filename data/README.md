# data/

cryoSPARC exports, one folder per dataset (same names as `results_cryodrgn/`), one subfolder per cryoSPARC job:
`heteroref_<J>` (heterogeneous refinement: particles, per-class files, class volumes), `homoref_<J>` (homogeneous
refinement), `refine_<J>_<mask>`, `3dva_<J>` (3D variability), `best_maps`, `nu_volumes` (NU maps of exported groups).
File names are cryoSPARC's own. Each file exists once (no copies across folders); cryoSPARC's per-slot exports
(`_blob`, `_ctf`, ...) are kept as written. Not versioned (too large for git).

| Dataset | Folders |
|---|---|
| `J4001_gP25_WT` | `heteroref_J1442`, `heteroref_J1497`, `homoref_J4001` (the J4001 particles), `refine_J4078_mask`, `3dva_J4002`, `3dva_J3428`, `3dva_J624`, `best_maps`, `nu_volumes` |
| `J399_hP7_E1371Q_ATP` | `heteroref_J264` (+ `classes/`, `nu_volumes/`), `homoref_J264`, `homoref_J399` (the J399 particles), `3dva_J406` |
| `J207_hP12_IDOR4` | `heteroref_J207` |
| `J2708_eP30_F508del_E1371W` | `heteroref_J2708`, `3dva_J2025` |
| `J4624_eP37_WT_T2a` | `heteroref_J4624` |

Old names (before 2026-11-09): `gP25W6_J1442_J1497` + `gP25W17_J4078_mask` + top-level J1442 `.cs` -> `J4001_gP25_WT`;
`hP7W1_J264` + `hP7W2_J399` -> `J399_hP7_E1371Q_ATP`; `hP12W2_J207`, `eP30W12_J2708`, `eP37W32_J4624` (and its copy `J4624`)
-> the folders above. Never used for training, moved to `../cryoem-classification_archive/data/`: `gP25W6_J985`,
`hP7W1_J235` (older heterorefs of the same particles), `yP14W13_J3578`, `yP14W13_J3654` (stacks not on disk).
