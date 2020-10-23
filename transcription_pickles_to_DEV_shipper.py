from Cfg import Cfg
from glob import glob
from pload import pload
import sys, os, tarfile
from glob import glob
from pathlib import Path
from datetime import datetime
from fixup import fixup
import numpy as np
np.seterr(all='raise')

def toqenize(pred):
    return pred.strip().split(' ')

def transcription_pickles_to_DEV_shipper(language, release):
    C = Cfg('NIST', 16000, _language=language, _phase='dev', _release=release) 
    tfns=glob(f'transcriptions/{language}/dev/{release}/*.pkl')
    results=[(fixup(fn), pload(fn)) for fn in tfns]
    ctms={fixup(fn):[] for fn in tfns}
    for fn,transcription in results:
        key=fn.split('_')
        ctm='_'.join(key[0:7])
        F='_'.join(key[0:6])
        channel=key[6]
        for tstart, tend, pred in list(sorted(transcription)):
            pred=pred.strip()
            if len(pred)==0:
                continue
            tbeg=tstart
            tdur=(tend-tstart)
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

        # import pandas as pd
        # pd.DataFrame(ctms[ctm],columns=['ctm','channel','start','duration','pred'])
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
    tar_fn=f'../../catskills_openASR20_{C.phase}_{C.language}_{C.release}.tgz'
    with tarfile.open(tar_fn, "w:gz") as tar: 
        for fn in glob('*.ctm'): 
            tar.add(fn)
    os.chdir('../../..')
    print('wrote', tar_fn)

if __name__=="__main__":
    import sys
    language=sys.argv[1]
    release=sys.argv[2]
    transcription_pickles_to_DEV_shipper(language, release)
