#!/bin/bash
# (11-25) pull compare JSONs + solid renders of the four datasets from hudson into deck/data
D=$(dirname $(readlink -f $0))/data; C=/data/hCFTR/mae2183/cryoem-classification/results_cryodrgn/model_pipeline/od/cmp; mkdir -p $D/renders
ssh -o ControlPath=$HOME/.ssh/cm-%r@%h -o BatchMode=yes mae2183@hudson.biology.columbia.edu "cd $C && tar -cf - compare_J4001.json compare_J207.json compare_J2708.json compare_J399.json -C rend/renders \$(cd rend/renders && ls solid_J4001_* solid_J207_* solid_J2708_* solid_J399_*)" | tar -C $D -xf - && mv -f $D/solid_*.png $D/renders/ 2>/dev/null; ls $D/renders | wc -l
