#!/usr/bin/bash -x
export language=vietnamese
export run=b04
#python splitter_transcriber.py $language $run 0 build 20
#python pred_pickles_to_shipper.py $language build $run
python run_scoring.py $language build $run
