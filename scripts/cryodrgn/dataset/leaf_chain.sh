#!/usr/bin/env bash
# Queue one leaf's import -> ab initio -> NU (jobs from cs_leaf_jobs.py STATE) as each parent completes, GPU jobs on GPU.
# Usage: leaf_chain.sh STATE LEAF_NAME GPU LOG
STATE=$1; TAG=$2; G=$3; LOG=$4
source "$(dirname "$(readlink -f "$0")")/lib.sh"
jg() { python3 -c "import json;d=json.load(open('$STATE'));print(d['project'] if '$1'=='project' else d['leaves']['$TAG']['$1'])"; }
P=$(jg project)
cs_run $P $(jg import_particles) && cs_run $P $(jg abinit) $G && cs_run $P $(jg nu) $G && log "done"
