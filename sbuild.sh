#!/usr/bin/bash -x
export phase=build
export release=b30
python pred_pickles_to_shipper.py vietnamese $phase $release
#python pred_pickles_to_shipper.py amharic $phase $release
#python pred_pickles_to_shipper.py pashto $phase $release
