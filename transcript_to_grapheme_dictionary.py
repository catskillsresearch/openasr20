#!/usr/bin/env python
# coding: utf-8

# Transcript to grapheme dictionary and training file

import os, json
from glob import glob

def transcript_to_grapheme_dictionary(C):
    language=C.language
    stage=C.stage
    transcripts = list(sorted(glob(f'{stage}/openasr20_{language}/build/transcription_split/*.txt')))
    T=''.join([open(transcript_file,'r').read().replace('\n','') for transcript_file in transcripts])
    vocab=list(set(''.join(T)))
    with open(C.grapheme_dictionary_fn,'w', encoding='utf-8') as f:
        json.dump(vocab, f)
    print('saved', C.grapheme_dictionary_fn)
