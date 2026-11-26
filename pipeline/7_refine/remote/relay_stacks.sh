#!/bin/bash
# (11-25) relay 128 px stacks hudson -> ganges / east through Argon pipes (no local copy); full md5 on both ends; .tmp -> final name only when identical.
# A complete-size .tmp left by an interrupted run is only re-checked, not re-sent.
H=mae2183@hudson.biology.columbia.edu; R=/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn
X() { timeout ${T:-7200} ssh -o ControlPath=$HOME/.ssh/cm-%r@%h -o BatchMode=yes "$@"; }
declare -A SRC=([J4001]=$R/J4001_gP25_WT [J207]=$R/J207_hP12_IDOR4 [J2708]=$R/J2708_eP30_F508del_E1371W [J399]=$R/J399_hP7_E1371Q_ATP)
for job in ${JOBS:-east:J207 ganges:J4001 ganges:J399 east:J2708 east:J399}; do h=${job%%:*}; d=${job#*:}; case $h in ganges) D=/data/mae2183_cs/mp/data/$d;; east) D=/scratch/mae2183/mp/data/$d;; esac
  t0=$(date +%s); M=mae2183@$h.biology.columbia.edu; S=${SRC[$d]}/inputs/particles.128.mrcs
  X $M "test -s $D/particles.128.mrcs" && { echo "$h $d already there"; continue; }
  n=$(X $H "stat -c %s $S"); m=$(X $M "stat -c %s $D/particles.128.mrcs.tmp 2>/dev/null")
  [ "$n" = "$m" ] || X $H "cat $S" | X $M "mkdir -p $D && cat > $D/particles.128.mrcs.tmp"
  a=$(X $H "md5sum $S" | cut -c1-32); b=$(X $M "md5sum $D/particles.128.mrcs.tmp" | cut -c1-32)
  if [ -n "$a" ] && [ "$a" = "$b" ]; then X $M "mv $D/particles.128.mrcs.tmp $D/particles.128.mrcs"; echo "$(date '+%H:%M') $h $d OK md5 $a $(( $(date +%s) - t0 )) s"
  else echo "$(date '+%H:%M') $h $d MD5 MISMATCH $a vs $b"; X $M "rm -f $D/particles.128.mrcs.tmp"; fi
done; echo RELAY_DONE
