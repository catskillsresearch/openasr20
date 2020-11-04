#!/usr/bin/bash -x
export language=vietnamese
export phase=dev
export release=400_unconstrained_afterburner
export maxduration=12
python translator.py $language $phase $release $maxduration
#python pred_pickles_to_shipper_with_afterburner.py $language $phase $release
#python run_scoring.py $language ${phase} ${release}
