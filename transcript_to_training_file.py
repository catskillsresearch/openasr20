#!/usr/bin/env python
# coding: utf-8

# # Transcript to grapheme dictionary and training file

import random, os
from glob import glob
stage='NIST'
language=os.getenv('language')
chunks = list(sorted(glob(f'{stage}/openasr20_{language}/build/transcription_split/*.txt')))
print(len(chunks), 'chunks')

L=[]
for text in chunks:
    audio=text.replace('transcription','audio').replace('txt', 'wav')
    L.append(f'{audio},{text}')

random.shuffle(L)

N=len(L)
train = L[0:int(N*0.9)]
val = L[len(train):]

#train = L[0:int(N*0.8)]
#val = L[len(train):len(train) + int(N*0.1)]
#test = L[len(train)+len(val):]

cases = [('train', train), ('valid', val)] # , ('test', test)]

for fn, L in cases:
    man_fn=f'analysis/{language}/{language}_{fn}.csv'
    with open(man_fn,'w') as f:
        f.write('\n'.join(L))
    print('saved', man_fn, 'with', len(L), 'examples')
