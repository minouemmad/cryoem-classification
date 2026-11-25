#!/usr/bin/env bash
# Run feeder_tick.py every 2 min until no chain job is building or active.   Log: $WORK/refine/feeder.log
source "$(dirname "$(readlink -f "$0")")/../../config/env.sh"; L=$WORK/refine/feeder.log
while true; do
  r=$(cs $ROOT/pipeline/7_refine/feeder_tick.py "'STATES': '$WORK/refine/cs_*.json', 'HOST': '$CS_HOST', 'GPU': $CS_GPU"); echo "$(date +%m-%d_%H:%M) $r" >> $L
  echo "$r" | grep -qE "'(building|queued|launched|started|running|waiting)'" || break; sleep 120
done
echo "$(date) feeder done" >> $L
