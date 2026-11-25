#!/usr/bin/env bash
# Hands-off K selection and refinement, every 10 min until all listed runs are handled:
#   reference run DONE -> kselect.py (label-free K = 2..8 scan) -> submit.sh with the candidate K set (ks=auto) or the listed K;
#   seed run DONE      -> repro.py (classes + rounds across seeds), kselect.py again (adds the seed columns), submit the listed K if any.
# Usage: auto_select.sh "DS/RUN:KS ..."   e.g. "J399/rae_base:auto J399/rae_s7:3 J399/rae_s101:"   (log $WORK/select/auto.log)
source "$(dirname "$(readlink -f "$0")")/../config/env.sh"; H=$ROOT/pipeline; O=$WORK/select; mkdir -p $O; L=$O/auto.log
log() { echo "$(date '+%m-%d %H:%M') $*" >> $L; }
while true; do
  left=0
  for t in $1; do r=${t%%:*}; ks=${t#*:}; ds=${r%%/*}; run=${r#*/}; [ -f $O/handled_${ds}_$run ] && continue
    [ -f $WORK/pipe/$r/DONE ] || { left=$((left + 1)); continue; }
    [ $run != rae_base ] && { log "repro $ds"; OMP_NUM_THREADS=4 $PY $H/6_classes/repro.py $ds > $O/repro_$ds.log 2>&1; }
    OMP_NUM_THREADS=4 $PY $H/6_classes/kselect.py $ds > $O/kselect_$ds.log 2>&1
    [ "$ks" = auto ] && ks=$($PY -c "import json; print(' '.join(map(str, json.load(open('$O/kselect_$ds.json'))['candidates'])))")
    [ -n "$ks" ] && { log "submit $r K $ks"; bash $H/7_refine/submit.sh $ds $run $ks; }
    touch $O/handled_${ds}_$run
  done
  [ $left = 0 ] && break; sleep 600
done
log "auto_select done"
