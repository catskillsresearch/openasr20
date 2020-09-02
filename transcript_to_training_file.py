#!/usr/bin/env python
# coding: utf-8

# # Transcript to grapheme dictionary and training file

import random
from glob import glob
stage='NIST'

chunks = list(sorted(glob(f'{stage}/*/build/transcription_split/*.txt')))

L=[]
for text in chunks:
    audio=text.replace('transcription','audio').replace('txt', 'wav')
    L.append(f'{audio},{text}')

random.shuffle(L)

N=len(L)
train = L[0:int(N*0.8)]
val = L[len(train):len(train) + int(N*0.1)]
test = L[len(val):]

train='\n'.join(train)
val='\n'.join(val)
test='\n'.join(test)

cases = [('train', train), ('valid', val), ('test', test)]
for fn, L in cases:
    man_fn=f'am_{fn}.csv'
    with open(man_fn,'w') as f:
        f.write(L)
