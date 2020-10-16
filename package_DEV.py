# coding: utf-8

import sys, os, tarfile
from glob import glob
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
np.seterr(all='raise')

def package_DEV(C, files, translations):
    ctms={'_'.join(os.path.basename(fn.split(',')[0]).split('_')[0:7]): [] for fn in files}
    for fn,pred in zip(files,translations):
        pred=pred.strip()
        if len(pred)==0:
            continue
        key=os.path.basename(fn)[0:-4].split('_')
        ctm='_'.join(key[0:7])
        F='_'.join(key[0:6])
        channel=key[6]
        tstart=float(key[-2])
        tend=float(key[-1])
        tbeg=tstart/C.sample_rate
        tdur=(tend-tstart)/C.sample_rate
        chnl='1' if channel=='inLine' else '2'
        tokens=pred.split(' ')
        n_tokens=len(tokens)
        token_lengths=np.array([len(token) for token in tokens])
        sum_token_lengths=token_lengths.sum()
        token_weights=token_lengths/sum_token_lengths
        dt=tdur*token_weights
        ends = tdur*np.cumsum(token_weights)
        tgrid=(ends-ends[0])+tbeg
        token_tstart=list(zip(tokens,tgrid))
        if ctms[ctm]: start_from = ctms[ctm][-1][2]
        for token, tstart, dt in zip(tokens,tgrid,dt):
            if token and token[0] not in ['(', '<']:
                row=(F,chnl,tstart,dt,token)
                ctms[ctm].append(row)
    for ctm in ctms:
       ctms[ctm].sort()
    shipping_dir=f'ship/{C.language}/{C.release}'
    os.system(f'mkdir -p {shipping_dir}')
    Path(shipping_dir).mkdir(parents=True, exist_ok=True)
    timestamp=datetime.today().strftime('%Y%m%d_%H%M')
    for ctm in ctms:
       fn=f'{C.shipping_dir}/{ctm}.ctm'
       with open(fn,'wt', encoding='utf-8') as f:
           for row in ctms[ctm]:
               line='\t'.join([str(x) for x in row])
               f.write(f"{line}\n")
    os.chdir(shipping_dir)
    tar_fn=f'../../catskills_openASR20_dev_{C.language}_{C.release}.tgz'
    with tarfile.open(tar_fn, "w:gz") as tar: 
        for fn in glob('*.ctm'): 
            tar.add(fn)
    os.chdir('../../..')
    print('wrote', tar_fn)
