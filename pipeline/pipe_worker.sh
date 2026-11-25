#!/bin/bash
# Pull run_model.sh jobs from $WORK/pipe_queue.txt (lines: DS NAME MODELDIR EPOCH [OPTS]; '#' = comment) until none is left and
# $WORK/pipe_queue.closed exists.  A line waits until MODELDIR/weights.EPOCH.pkl exists (models trained elsewhere are picked up when copied in).
# Usage: pipe_worker.sh GPU   (one per GPU)
source "$(dirname "$(readlink -f "$0")")/../config/env.sh"; Q=$WORK/pipe_queue.txt; GPU=$1
next() { ( flock 9; while read -r ds nm md ep opts; do [[ -z $ds || $ds == \#* ]] && continue; grep -qxF "$ds/$nm" $WORK/pipe_taken.txt 2>/dev/null && continue
           [ -f $md/weights.$ep.pkl ] || continue; echo "$ds/$nm" >> $WORK/pipe_taken.txt; echo "$ds $nm $md $ep $opts"; break; done < $Q ) 9>$Q.lock; }
while true; do j=$(next); if [ -z "$j" ]; then [ -f $WORK/pipe_queue.closed ] && break; sleep 600; continue; fi; bash $ROOT/pipeline/run_model.sh $GPU $j; done
echo "$(date '+%m-%d %H:%M') [gpu$GPU] pipeline worker exits" >> $WORK/pipe.log
