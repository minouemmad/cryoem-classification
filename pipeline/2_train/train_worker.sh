#!/bin/bash
# Training queue worker (any machine; one per GPU).  Usage: train_worker.sh CONF GPU [MAXWIDTH]
#   CONF (see config/machine.conf.example) sets BASE (queue.txt, taken.txt, models/<DS>/<NAME>), DATA (<DS>/particles.128.mrcs, poses.pkl, ctf.pkl), PY, CRYODRGN, SCD (this dir).
#   Queue line: TYPE DS NAME SEED ZDIM BETA LAMBDA [WIDTH=1024] [EPOCHS=50] [STACK=particles.128.mrcs]
#     TYPE rae (train_rae.py, --rae-beta / --rae-lambda) | vae (cryodrgn train_vae, --beta) | ae (train_ae.py, plain autoencoder)
#   A job is taken only when DATA/DS/STACK.ok exists (stack fully copied) and WIDTH <= MAXWIDTH (e.g. 256 on 11 GB cards).
#   Fixed: NU poses, encoder/decoder WIDTH x 3, batch 64, lazy loading, fp32, checkpoint every 5 epochs; resumes from the latest checkpoint.
source $1; GPU=$2; MAXW=${3:-99999}; NW=${NW:-4}; export CUDA_DEVICE_ORDER=PCI_BUS_ID; Q=$BASE/queue.txt
log() { echo "$(date '+%m-%d %H:%M') [${TAG:-$(hostname -s)} gpu$GPU] $*" >> $BASE/worker.log; }
next() { ( flock 9; while read -r t ds nm rest; do [[ -z $t || $t == \#* ]] && continue; grep -qxF "$ds/$nm" $BASE/taken.txt 2>/dev/null && continue
           set -- $rest; [ ${5:-1024} -le $MAXW ] && [ -f $DATA/$ds/${7:-particles.128.mrcs}.ok ] || continue
           echo "$ds/$nm" >> $BASE/taken.txt; echo "$t $ds $nm $rest"; break; done < $Q ) 9>$Q.lock; }
while j=$(next); [ -n "$j" ]; do
  read t ds nm seed z beta lam width ep stk <<< "$j"; width=${width:-1024}; ep=${ep:-50}; stk=${stk:-particles.128.mrcs}
  O=$BASE/models/$ds/$nm; D=$DATA/$ds; mkdir -p $O; [ -f $O/weights.$ep.pkl ] && { log "skip $ds/$nm (done)"; continue; }
  LOAD=; ls $O/weights.*.pkl > /dev/null 2>&1 && LOAD="--load latest"
  A="$D/$stk --poses $D/poses.pkl --ctf $D/ctf.pkl --zdim $z --enc-dim $width --enc-layers 3 --dec-dim $width --dec-layers 3 --no-amp -n $ep --lazy
     --num-workers $NW --batch-size 64 --checkpoint 5 --log-interval 5000 --seed $seed --no-analysis $LOAD -o $O"
  log "train $t $ds/$nm seed $seed z $z beta $beta lambda $lam width $width epochs $ep stack $stk $LOAD"
  case $t in
    rae) (cd $SCD && CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 $PY train_rae.py $A --rae-beta $beta --rae-lambda $lam >> $O/run.log 2>&1);;
    ae)  (cd $SCD && CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 $PY train_ae.py $A >> $O/run.log 2>&1);;
    *)   CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 $CRYODRGN train_vae $A --beta $beta >> $O/run.log 2>&1;;
  esac
  [ -f $O/weights.$ep.pkl ] && { touch $O/DONE; log "done $ds/$nm"; } || log "FAILED $ds/$nm (see $O/run.log)"
done; log "queue empty, worker exits"
