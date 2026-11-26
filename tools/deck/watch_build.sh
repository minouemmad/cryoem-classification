#!/bin/bash
# (11-25) keep the deck current: every 30 min pull compare JSONs + renders from hudson; rebuild when they changed (build to .tmp, then move into place).
# Final build (with preview) once hudson od/cmp/finish_all.done exists (dispatcher done + hudson feeder done + last compare pass).
D=$(dirname $(readlink -f $0)); OUT=/Dedicated/schnieders/maemmad/cryoem-classification/results_cryodrgn/CFTR_label_free_classes_five_starts.pptx; last=""
while true; do
  fin=0; ssh -o ControlPath=$HOME/.ssh/cm-%r@%h -o BatchMode=yes mae2183@hudson.biology.columbia.edu "test -f /data/hCFTR/mae2183/cryoem-classification/results_cryodrgn/model_pipeline/od/cmp/finish_all.done" && fin=1
  if $D/fetch.sh > /dev/null; then
    sig=$(cat $D/data/compare_*.json | md5sum | cut -c1-32)
    if [ "$sig" != "$last" ] || [ $fin = 1 ]; then
      cd $D && python3 build_deck.py data ${OUT%.pptx}.tmp.pptx $( [ $fin = 1 ] && echo --preview ) > $D/build.log 2>&1 && mv -f ${OUT%.pptx}.tmp.pptx $OUT && last=$sig \
        && echo "$(date '+%m-%d %H:%M') DECK_UPDATED $(python3 -c "import json,glob; print(' '.join('%s %d maps' % (f.split('_')[-1][:-5], sum(1 for r in json.load(open(f))['rows'] if r['run'].startswith('start_'))) for f in sorted(glob.glob('$D/data/compare_*.json'))))")"
    fi
  fi
  [ $fin = 1 ] && { echo "DECK_FINAL $OUT"; break; }
  sleep 1800
done
