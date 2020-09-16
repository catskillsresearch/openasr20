# coding: utf-8

import os
from glob import glob
import json
from tqdm import tqdm

def trim_to_max_samples_per_word(C):
    cfn=f'analysis/{C.language}/{C.language}_samples_per_word_cutoff_95pct.json'
    with open(cfn,'r') as f:
        cutoff=json.load(f)['samples_per_word_cutoff_95pct']
    lengths=[]
    transcripts = list(sorted(glob(f'{C.stage}/openasr20_{C.language}/build/transcription_split/*.txt')))
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

