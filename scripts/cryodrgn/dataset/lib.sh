# Shared shell helpers for the dataset pipeline (source this file). Callers set LOG (and optionally TAG).
ROOT=/data/hCFTR/mae2183/cryoem-classification
SC=$ROOT/scripts/cryodrgn; RES=$ROOT/results_cryodrgn; LOCKS=$RES/.gpu_locks
PY=~/miniconda3/envs/cryodrgn/bin/python; CD=~/miniconda3/envs/cryodrgn/bin/cryodrgn
CSM=/opt/cryoem/cryosparc_master/bin/cryosparcm

log() { echo "$(date '+%m-%d %H:%M') ${TAG:+$TAG }$*" >> "$LOG"; }
jget() { python3 -c "import json;print(json.load(open('$1'))['$2'])"; }

cs_status() { $CSM cli "api.jobs.get_status('$1','$2').value" 2>/dev/null | tail -n 1 | tr -d "'\""; }

cs_run() {   # PROJECT JOB [GPU]: queue the job if still building, then wait; non-zero if it fails or is killed
  local p=$1 j=$2 g=${3:-} s
  if [ "$(cs_status $p $j)" = building ]; then
    $CSM cli "api.jobs.enqueue('$p','$j', hostname='hudson'${g:+, gpus=[$g]})" > /dev/null 2>&1
    log "enqueued $p $j"
  fi
  while true; do
    s=$(cs_status $p $j)
    [ "$s" = completed ] && { log "$p $j completed"; return 0; }
    case "$s" in failed|killed) log "$p $j $s - stopping"; return 1;; esac
    sleep 120
  done
}

gpu_lock() {   # OWNER: set GPU to the first GPU (3 2 0 1) with < 40 GB in use and a free lock; wait if none
  GPU=""; mkdir -p $LOCKS
  while [ -z "$GPU" ]; do
    for g in 3 2 0 1; do
      mem=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
      if [ "$mem" -lt 40000 ] && mkdir $LOCKS/gpu$g 2>/dev/null; then GPU=$g; echo "$1" > $LOCKS/gpu$g/owner; break; fi
    done
    [ -z "$GPU" ] && sleep 600
  done
  trap "rm -rf $LOCKS/gpu$GPU" EXIT
}

class_set() {   # NAME -> recon_tree --class-set arguments from prepare.py's prep.json
  python3 -c "
import json; p = json.load(open('$RES/$1/nu_consensus/prep.json'))
print(p['name'].split('_')[0], ' '.join(f'{c}={f}' for c, f in zip(p['class_counts'], p['class_files'])))"
}

run_tree() {   # NAME AXIS(pc1|latent) KRULE(incremental|best): recon tree on the pipeline's decoded centres, on $GPU
  local R=$RES/$1 sp=volume sfx=$3 O
  [ $2 = latent ] && sp=latent
  [ $3 = best ] && sfx=bestK
  O=$R/recon_tree/tree_${sp}_pc1_$sfx
  log "tree ($2, $3 K) -> $O"
  CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 $PY $SC/recon_tree.py --decoded $R/recon_tree/decoded2000_nuposes_e50 \
    --mask64 $R/recon_tree/mask64.mrc --inputs $R/inputs --particles-cs $R/nu_consensus/$1_particles.cs --class-set $(class_set $1) \
    --accept cv --cv-rule corrected --ks 2 3 4 5 6 7 8 --k-rule $3 --axis $2 -o $O > $O.log 2>&1 \
    && log "tree done ($2, $3 K): $(tail -n 1 $O.log)" || { log "tree failed ($2, $3 K)"; return 1; }
}
