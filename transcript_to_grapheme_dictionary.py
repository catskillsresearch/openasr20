#!/usr/bin/env python
# coding: utf-8

# Transcript to grapheme dictionary and training file

from glob import glob
stage='NIST'
transcripts = list(sorted(glob(f'{stage}/*/build/transcription/*.txt')))
len(transcripts)

import os
import pandas as pd
from txt_to_stm import txt_to_stm

T=[]
for transcript_file in transcripts:
    file = "_".join(os.path.basename(transcript_file).split("_")[:-1])
    channel = os.path.basename(transcript_file).split("_")[-1].split(".")[-2]
    transcript_df = pd.read_csv(transcript_file, sep = "\n", header = None, names = ["content"])
    result = [x[-1] for x in txt_to_stm(transcript_df, file, channel) if len(x)==6]
    T.append(''.join(result))

vocab=list(set(''.join(T)))

import json
with open('amharic_characters.json','w') as f:
    json.dump(vocab, f)
