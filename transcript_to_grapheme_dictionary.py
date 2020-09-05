#!/usr/bin/env python
# coding: utf-8

# Transcript to grapheme dictionary and training file

import os, json
from glob import glob

language=os.getenv('language')
stage='NIST'
transcripts = list(sorted(glob(f'{stage}/openasr20_{language}/build/transcription_split/*.txt')))
T=''.join([open(transcript_file,'r').read().replace('\n','') for transcript_file in transcripts])
vocab=list(set(''.join(T)))
fn = f'{language}_characters.json'
with open(fn,'w', encoding='utf-8') as f:
    json.dump(vocab, f)
print('saved', fn)
