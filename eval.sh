#!/usr/bin/bash -x
export language=somali
export phase=eval
export release=002_constrained
export maxduration=10
date > start_eval.txt
python translator.py $language $phase $release $maxduration
date > finish_eval.txt
python pred_pickles_to_shipper.py $language $phase $release
