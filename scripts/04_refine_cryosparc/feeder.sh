#!/usr/bin/env bash
# Run feeder_tick.py every 2 min until none of the centre-scan jobs is building or active.
SCR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
S=/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn/J4001_gP25_WT/candidate_groups_epoch50/axis_survey
while true; do
  r=$(/opt/cryoem/cryosparc_master/bin/cryosparcm cli "(lambda g: (exec(open('$SCR/04_refine_cryosparc/feeder_tick.py').read(), g), g['RESULT'])[1])({'api': api})" 2>&1 | tail -n 1)
  echo "$(date +%m-%d_%H:%M) $r" >> $S/feeder_kc.log
  if ! echo "$r" | grep -qE "'(building|queued|launched|started|running|waiting)'"; then break; fi
  sleep 120
done
echo "$(date) feeder_kc done" >> $S/feeder_kc.log
