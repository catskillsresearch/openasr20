#!python
# coding: utf-8

import os
from glob import glob

def make_DEV_infer_csv(C):
    split_files = '\n'.join([f'{x},infer.txt' for x in list(sorted(glob(f'{C.stage}/openasr20_{C.language}/dev/audio_split/*.wav')))])
    fn = f'analysis/{C.language}/DEV_{C.language}_split.csv'
    with open(fn, 'w') as f:
        f.write(split_files)
    print('saved', fn)
