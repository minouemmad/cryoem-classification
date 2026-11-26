#!/bin/bash
# (11-25) Argon loop: every 30 min score the J4001 axis windows on hudson; queue stage-2 (size) sets as soon as a stage-1 axis is complete; stop when every stage 2 is scored.
A=/Dedicated/schnieders/maemmad/cryoem-classification/results_cryodrgn/argon_dispatch; B=/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn/model_pipeline; PY=/home/mae2183/miniconda3/envs/cryodrgn/bin/python
SSH="ssh -o ControlPath=$HOME/.ssh/cm-%r@%h -o BatchMode=yes mae2183@hudson.biology.columbia.edu"
while true; do
  out=$($SSH "cd $B && OMP_NUM_THREADS=2 nice $PY od/analysis/win_score.py 2>&1 | grep -v Warn"); echo "$(date '+%m-%d %H:%M')"; echo "$out"
  echo "$out" | grep '^STAGE2' | while read -r _ DS RUN J C; do
    r=$($SSH -n "cd $B && $PY od/analysis/win_build.py $DS $RUN $J stage2 $C" | tail -1); nm=$(echo $r | awk '{print $1}')
    [ -n "$nm" ] && ! grep -q "^$DS $nm " $A/extra_sets.txt && echo "$DS $nm 6" >> $A/extra_sets.txt && echo "QUEUED $DS $nm ($r)"
  done
  n2=$(echo "$out" | grep -c 'pick sd'); [ "$n2" -ge 4 ] && { echo WINDOWS_DONE; break; }
  sleep 1800
done
