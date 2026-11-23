#!/usr/bin/env bash
# after residual_split.py: export the two halves and create the cryoSPARC chains (import -> ab initio -> NU, project P17 / W1,...
SCR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
R=/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn; A=$R/J4001_gP25_WT/candidate_groups_epoch50/axis_survey; RV=$R/J4001_gP25_WT/method_rank/rv; PY=/home/mae2183/miniconda3/envs/cryodrgn/bin/python; DS=$1; TAG=$2; CS=/opt/cryoem/cryosparc_master/bin/cryosparcm
if [ "$DS" = J4001 ]; then
  NM=$(echo $TAG | tr -d '_.'); $PY $SCR/04_refine_cryosparc/export_groups.py $A/export_manifest_set_rv_$TAG.json $A/set_rv_$TAG J4001rv$NM leaves:2 >> $RV/rv_submit.log 2>&1 || { echo "export FAILED $TAG" >> $RV/rv_submit.log; exit 1; }
  $CS cli "(lambda g: (exec(open('$SCR/04_refine_cryosparc/submit_chains.py').read(), g), g['RESULT'])[1])({'api': api, 'PROJ': 'P17', 'WS': 'W1', 'GPUS': [2], 'REF_PROJ': 'P17', 'REF_AB': 'J373', 'REF_NU': 'J374', 'MANIFEST': '$A/export_manifest_set_rv_$TAG.json', 'STATE': '$A/cryosparc_jobs_kcP17_f_rv_$TAG.json'})" 2>&1 | tail -1 >> $RV/rv_submit.log
else
  J=$R/J4624_eP37_WT_T2a/recon_tree; $PY $SCR/04_refine_cryosparc/star_local_stack.py V$TAG >> $RV/rv_submit.log 2>&1 || { echo "star FAILED $TAG" >> $RV/rv_submit.log; exit 1; }
  $CS cli "(lambda g: (exec(open('$SCR/04_refine_cryosparc/submit_chains_local_stack.py').read(), g), g['RESULT'])[1])({'api': api, 'PROJ': 'P17', 'WS': 'W1', 'GPUS': [2], 'REF_PROJ': 'P17', 'REF_AB': 'J451', 'REF_NU': 'J452', 'STARS': ['$J/rs_setV$TAG/cryosparc/eP37_rsV${TAG}d128_leaf0.star', '$J/rs_setV$TAG/cryosparc/eP37_rsV${TAG}d128_leaf1.star'], 'STATE': '$A/cryosparc_jobs_kcP17_f_j4624rv_$TAG.json'})" 2>&1 | tail -1 >> $RV/rv_submit.log
fi
pgrep -f "04_refine_cryosparc/feeder.sh" > /dev/null || (setsid nohup bash $SCR/04_refine_cryosparc/feeder.sh > /dev/null 2>&1 < /dev/null &)
echo "$(date '+%m-%d %H:%M') submitted $DS $TAG" >> $RV/rv_submit.log
