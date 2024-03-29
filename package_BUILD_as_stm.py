#!/usr/bin/env python
# coding: utf-8

import sys
from glob import glob
import os, tarfile
from pathlib import Path
from datetime import datetime

def package_build_as_stm():
    try:
        release=os.getenv('release_number')
        if not release:
            release = 0
    except:
        release = 0

    phase='build'
    model='graphemes'
    sample_rate=8000
    stage='NIST'

    with open(translation_file, 'r', encoding='utf-8') as f:
        translations=f.readlines()
    print('read', translation_file)

    split_files = list(sorted(glob(f'{stage}/openasr20_{language}/build/transcription_split/{recording}*.txt')))

    ctms={'_'.join(os.path.basename(fn.split(',')[0]).split('_')[0:7]): [] for fn in split_files}
    import numpy as np
    import pandas as pd
    for fn,pred in zip(split_files,translations):
        key=os.path.basename(fn)[0:-4].split('_')
        ctm='_'.join(key[0:7])
        F='_'.join(key[0:6])
        channel=key[6]
        tbeg=float(key[8])
        tdur=float(key[9])-tbeg
        chnl='1' if channel=='inLine' else '2'
        if ctms[ctm]:
            start_from = ctms[ctm][-1][2]
        ctms[ctm].append((F,chnl,tbeg,tdur,pred))
        ctms[ctm].sort()

    os.system(f'mkdir -p analysis/{language}/build_inference')

    for ctm in ctms:
        fn=f'analysis/{language}/build_inference/{ctm}.stm'
        with open(fn,'wt', encoding='utf-8') as f:
            for row in ctms[ctm]:
                line='\t'.join([str(x) for x in row])
                f.write(f"{line}\n")
        print('wrote', fn)

language, recording, translation_file = sys.argv[1:]
guarani   kurmanji-kurdish   tamil cantonese  javanese  mongolian         somali  vietnamese
