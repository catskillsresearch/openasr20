#!/usr/bin/env python
# coding: utf-8

from glob import glob
import os, sys, tarfile
from pathlib import Path
from datetime import datetime

try:
    release=os.getenv('release_number')
    if not release:
        release = 0
except:
    release = 0

phase='dev'
model='graphemes'
sample_rate=8000
language=os.getenv('language')
stage='NIST'
translation_file=f'analysis/{language}/RESULT_{language}_trimmed.txt'

with open(translation_file, 'r', encoding='utf-8') as f:
    translations=f.readlines()

split_files = [f'{x},infer.txt' for x in list(sorted(glob(f'{stage}/openasr20_{language}/dev/audio_split/*.wav')))]

ctms={'_'.join(os.path.basename(fn.split(',')[0]).split('_')[0:7]): [] for fn in split_files}

for fn,pred in zip(split_files,translations):
    key=os.path.basename(fn.split(',')[0])[0:-4].split('_')
    ctm='_'.join(key[0:7])
    F='_'.join(key[0:6])
    channel=key[6]
    tstart=float(key[8])
    tend=float(key[9])
    tbeg=tstart/sample_rate
    tdur=(tend-tstart)/sample_rate
    chnl='1' if channel=='inLine' else '2'
    tokens=pred[0:-1].split(' ')
    n_tokens=len(tokens)
    dt = tdur/n_tokens
    tgrid=[tdur+i*dt for i in range(n_tokens)]
    token_tstart=list(zip(tokens,tgrid))
    for token, tstart in token_tstart:
        if token and token[0] not in ['(', '<']:
            row=(F,chnl,tstart,dt,token)
            ctms[ctm].append(row)

shipping_dir=f'ship/{language}/{release}'
Path(shipping_dir).mkdir(parents=True, exist_ok=True)

timestamp=datetime.today().strftime('%Y%m%d_%H%M')

for ctm in ctms:
    fn=f'{shipping_dir}/{ctm}.ctm'
    with open(fn,'wt', encoding='utf-8') as f:
        for row in ctms[ctm]:
            line='\t'.join([str(x) for x in row])
            f.write(f"{line}\n")

os.chdir(shipping_dir)

tar_fn=f'../../catskills_openASR20_{phase}_{language}_{release}.tgz'

with tarfile.open(tar_fn, "w:gz") as tar:
    for fn in glob('*.ctm'):
        tar.add(fn)

os.chdir('../..')

print('saved', tar_fn)
