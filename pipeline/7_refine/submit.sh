#!/usr/bin/env bash
# Classes of pipe/<DS>/<RUN> at each K (or RUN=hetero) -> STARs -> cryoSPARC chains (import -> ab initio -> NU; parameters of REF_AB / REF_NU,
# default P17 J451 / J452) in $CS_PROJECT, then start the feeder (one per machine) if it is not running.   Usage: submit.sh DS RUN K [K ...]
source "$(dirname "$(readlink -f "$0")")/../../config/env.sh"; H=$ROOT/pipeline/7_refine; O=$WORK/refine; mkdir -p $O; L=$O/submit.log
DS=$1; RUN=$2; shift 2; REF_AB=${REF_AB:-J451}; REF_NU=${REF_NU:-J452}
$PY $H/stars.py $DS $RUN "$@" >> $L 2>&1 || { echo "$(date '+%m-%d %H:%M') FAILED stars $DS $RUN" >> $L; exit 1; }
Ks="$*"; [ $RUN = hetero ] && Ks=$(ls -d $O/$DS/hetero_K* | sed 's/.*_K//')     # hetero: K = number of heterorefinement classes
for K in $Ks; do
  S=$(ls $O/$DS/${RUN}_K$K/cryosparc/*.star | sort -V | sed "s/.*/'&'/" | paste -sd,)
  r=$(cs $H/cs_chain.py "'PROJ': '$CS_PROJECT', 'WS': 'W1', 'HOST': '$CS_HOST', 'GPU': $CS_GPU, 'REF_PROJ': 'P17', 'REF_AB': '$REF_AB', 'REF_NU': '$REF_NU', 'STARS': [$S],
        'STATE': '$O/cs_${DS}_${RUN}_K$K.json', 'TITLE': '$DS fully conditioned search', 'DATASET': '$DS'")
  echo "$(date '+%m-%d %H:%M') $DS $RUN K$K: $r" >> $L
done
pgrep -f "7_refine/feeder.sh" > /dev/null || (setsid nohup bash $H/feeder.sh > /dev/null 2>&1 < /dev/null &)
