# Site paths, sourced by every shell script in pipeline/ (export any of them first to override).
# Python steps read the same variables, so `source config/env.sh` before running one by hand.
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${PY:-$HOME/miniconda3/envs/cryodrgn/bin/python}
CRYODRGN=${CRYODRGN:-$HOME/miniconda3/envs/cryodrgn/bin/cryodrgn}
RES=${RES:-/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn}   # per-dataset folders: <dataset>/nu_consensus, <dataset>/inputs
WORK=${WORK:-$RES/model_pipeline}                                       # models/, pipe/, env/, od/, queues and logs
DATASETS=${DATASETS:-$ROOT/config/datasets.tsv}
CSM=${CSM:-/opt/cryoem/cryosparc_master/bin/cryosparcm}
CS_HOST=${CS_HOST:-hudson}; CS_PROJECT=${CS_PROJECT:-P17}; CS_GPU=${CS_GPU:-2}
CS_PROJECT_DIR=${CS_PROJECT_DIR:-/data/hCFTR/mae2183/cs_projects/CS-j4001-rae-window-centre-scan-plateau-sizes}
CHIMERAX=${CHIMERAX:-/data/hCFTR/mae2183/tools/chimerax/usr/bin/chimerax}
CX_LD=${CX_LD:-$HOME/miniconda3/pkgs/mesalib-24.3.4-h0b126fc_1/lib:$HOME/miniconda3/pkgs/libllvm19-19.1.7-hf7376ad_2/lib:$HOME/miniconda3/lib}   # headless ChimeraX
# deck frame used by 8_compare (large maps, not in git): reference map, protein mask, camera map, core model for the refit
DECK=${DECK:-$RES/J4001_gP25_WT/candidate_groups_epoch50/presentation_v7}
DECK_REF=${DECK_REF:-/data/hCFTR/mae2183/cryoem-classification/data/gP25W6_J1442_J1497/J1497_classes/heteroref_volumes/cryosparc_P25_J1497_class_08_00000_volume_sharp.mrc}
export ROOT PY CRYODRGN RES WORK DATASETS CSM CS_HOST CS_PROJECT CS_GPU CS_PROJECT_DIR CHIMERAX CX_LD DECK DECK_REF
cx() { LD_LIBRARY_PATH=$CX_LD "$CHIMERAX" --nogui --offscreen --exit --script "$@"; }   # cx script.py
cs() { "$CSM" cli "(lambda g: (exec(open('$1').read(), g), g['RESULT'])[1])({'api': api, $2})" 2>&1 | tail -n 1; }   # cs script.py "'KEY': value, ..."
