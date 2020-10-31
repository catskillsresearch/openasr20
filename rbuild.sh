#!/usr/bin/bash -x
export phase=build
export release=b30
python run_scoring.py vietnamese ${phase} ${release}
#python run_scoring.py amharic ${phase} ${release}
#python run_scoring.py pashto ${phase} ${release}
