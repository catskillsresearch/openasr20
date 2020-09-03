#!/usr/bin/env python
# coding: utf-8

from glob import glob

stage='NIST'
split_files = '\n'.join([f'{x},infer.txt' for x in list(sorted(glob(f'{stage}/*/dev/audio_split/*.wav')))])

fn = 'DEV_split.csv'
with open(fn, 'w') as f:
    f.write(split_files)
print('saved', fn)
