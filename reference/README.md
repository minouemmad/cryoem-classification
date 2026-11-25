# Reference files

| Path | Content |
|---|---|
| `pdb/D1557_fitT.pdb` | V-shape CFTR model (inward-facing, NBDs apart) in the J4001 deck frame; core model for the TMD refit (residues 1-388, 841-1173). Best envelope fit for J4001 / J4624 |
| `pdb/6msm_protein.pdb`, `pdb/6MSM.cif` | human CFTR E1371Q, phosphorylated, ATP-bound (NBD dimer). Best envelope fit for J207 / J2708 / J399 |
| `pdb/5UAK.cif` | dephosphorylated, ATP-free human CFTR (fit check) |
| `pdb/deck_fit_*.json` | fit correlations of 6MSM / 5UAK and hand check in the deck frame |
| `envelopes/<DS>/` | `env64.npy` consensus envelope, `env64_pdb.npy` PDB envelope (default), `mask64.mrc`, `fit_out.txt` (both models x both hands), `regions.log`, `apix.txt` (64^3 voxel, A) |
| `cryosparc/consensus/` | consensus-pose chains per dataset (`cryosparc_job.json`) and `prep.json` (box, pixel size, heterorefinement class counts) |
| `cryosparc/class_refinements/` | job uids of every class-map chain (import -> ab initio -> NU) in hudson P17, one file per refined set |
| `results/` | `kselect_<DS>.json` (label-free K scan), `J4001_J4624_{stoprule,reprok,kcontent}.json` (seed-reproducible rounds / classes, K content) |

## cryoSPARC provenance

| Dataset | Consensus poses | Heterorefinement classes (evaluation only) |
|---|---|---|
| J4001 (gP25 WT, ATP) | NU J1049 (particles J824, volume J825) | P25 J1497 / J2923 labels (P6, P7, P8, Alt-NBD1) |
| J4624 (eP37 WT + T2a) | P5: import J29/J30 -> homo J31 -> backproject J32 -> NU J33 | C06-C10 |
| J207 (hP12 IDOR4) | P12: import J374/J375 -> homo J376 -> backproject J377 -> NU J378 | C06 35,067 · C07 25,973 |
| J2708 (eP30 F508del E1371W) | P6: import J27/J28 -> homo J29 -> backproject J30 -> NU J31 | 4 classes: 53,461 · 117,761 · 26,889 · 31,299 |
| J399 (hP7 E1371Q ATP) | P7 homo refine J399 -> NU J1314 (params J766); the RAE is trained on the J1314 poses | P7 J264 C06-C14: 66,431 · 57,928 · 45,279 · 50,279 · 42,739 · 17,347 · 8,027 · 8,502 · 3,109 |

New consensus chains copy their parameters from P7 J399 (homo), P7 J825 (backprojection), P7 J647 (NU).
Class maps (label-free and heterorefinement alike) are refined in hudson P17 with the parameters of P17 J451 (ab initio, 1 class) and P17 J452 (NU),
from STARs rewritten to the local 128 px stack (pixel size x box / 128), imported without poses.
In `class_refinements/`: `mc4` = fully conditioned search classes K = 3, `mc4k4` / `mc4k5` = K = 4 / 5, `mc4s7` = K = 3 of the seed-7 replica model.
