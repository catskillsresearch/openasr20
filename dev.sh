#!/usr/bin/bash -x
export language=vietnamese
export phase=dev
export release=u300
export maxduration=20
#python translator.py $language $phase $release $maxduration
python pred_pickles_to_shipper.py $language $phase $release
#python run_scoring.py $language ${phase} ${release}

