#!python
# coding: utf-8

import os
from glob import glob

stage='NIST'
language=os.getenv('language')
split_files = '\n'.join([f'{x},infer.txt' for x in list(sorted(glob(f'{stage}/openasr20_{language}/dev/audio_split/*.wav')))])

fn = f'DEV_{language}_split.csv'
with open(fn, 'w') as f:
    f.write(split_files)
print('saved', fn)
