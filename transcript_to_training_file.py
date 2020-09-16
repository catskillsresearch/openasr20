#!/usr/bin/env python
# coding: utf-8

# # Transcript to grapheme dictionary and training file

import random, os
from glob import glob

def transcript_to_training_file(C):
    chunks = list(sorted(glob(f'{C.stage}/openasr20_{C.language}/build/transcription_split/*.txt')))
    print(len(chunks), 'chunks')
    L=[]
    for text in chunks:
        audio=text.replace('transcription','audio').replace('txt', 'wav')
        L.append(f'{audio},{text}')
    random.shuffle(L)
    N=len(L)
    train = L[0:int(N*0.9)]
    val = L[len(train):]
    cases = [('train', train), ('valid', val)] # , ('test', test)]
    for fn, L in cases:
        man_fn=f'analysis/{C.language}/{C.language}_{fn}.csv'
        with open(man_fn,'w') as f:
            f.write('\n'.join(L))
        print('saved', man_fn, 'with', len(L), 'examples')
