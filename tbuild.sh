#!/usr/bin/bash -x
export phase=build
export release=b30
export maxduration=30
python translator.py vietnamese ${phase} ${release} $maxduration
#python translator.py amharic ${phase} ${release} $maxduration
#python translator.py pashto ${phase} ${release} $maxduration
