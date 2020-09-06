# coding: utf-8

import os
from glob import glob
import json
from tqdm import tqdm

language=os.getenv('language')
stage=f'NIST'
sample_rate=8000

cfn=f'{language}_samples_per_word_cutoff_95pct.json'
with open(cfn,'r') as f:
    cutoff=json.load(f)['samples_per_word_cutoff_95pct']

lengths=[]
transcripts = list(sorted(glob(f'{stage}/openasr20_{language}/build/transcription_split/*.txt')))
trimmed = 0
for transcript_file in tqdm(transcripts):
    with open(transcript_file,'r') as f:
        sentence=f.read().strip().split(' ')
        n_words = len(sentence)
    audio_file = transcript_file.replace('/transcription_split/', '/audio_split/').replace('.txt','.wav')
    start,end = audio_file.split('_')[-2:]
    start=float(start)*sample_rate
    end=float(end[0:-4])*sample_rate
    samples=end-start
    samples_per_word=samples/n_words
    if samples_per_word > cutoff:
        os.remove(transcript_file)
        os.remove(audio_file)
        trimmed += 1

if trimmed:
    print(f'trimmed {trimmed} files below sample rate {cutoff}')

