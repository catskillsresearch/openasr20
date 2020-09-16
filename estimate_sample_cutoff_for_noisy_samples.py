#!/usr/bin/env python
# coding: utf-8

#  # Word size versus segment length

import os
from glob import glob
from tqdm import tqdm
import librosa
import numpy as np
import json

def estimate_sample_cutoff_for_noisy_samples(C):
    language=C.language
    stage=C.stage
    sample_rate=C.sample_rate
    transcripts = list(sorted(glob(f'{stage}/openasr20_{language}/build/transcription_split/*.txt')))
    lengths=[]
    for transcript_file in tqdm(transcripts):
        with open(transcript_file,'r') as f:
            sentence=f.read().strip().split(' ')
            n_words = len(sentence)
        audio_file = transcript_file.replace('/transcription_split/', '/audio_split/').replace('.txt','.wav')
        x_np,sr=librosa.load(audio_file, sr=sample_rate)
        n_samples = x_np.shape[0]
        lengths.append((n_samples, n_words, sentence, transcript_file, audio_file))
    samples_per_word=np.array(list(sorted([x/y for x,y,z,t,a in lengths])))
    cutoff=samples_per_word[int(samples_per_word.shape[0]*0.95)]
    with open(C.sample_cutoff_fn, 'w') as f:
        json.dump({'samples_per_word_cutoff_95pct': cutoff}, f)
    print(f'cutoff for {language} is {cutoff} samples per word')
    print('saved', C.sample_cutoff_fn)
