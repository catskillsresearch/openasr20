#!/usr/bin/bash -x
export language=somali
export phase=dev
export release=final_constrained
export maxduration=10
export CUDA_VISIBLE_DEVICES=1
date > start_dev.txt
python translator.py $language $phase $release $maxduration
date > finish_dev.txt
python pred_pickles_to_shipper.py $language $phase $release
#python run_scoring.py $language ${phase} ${release}
