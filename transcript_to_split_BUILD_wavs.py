#!/usr/bin/env python
# coding: utf-8

# # Transcript to BUILD wavs

from glob import glob
import os
from matplotlib.pylab import *
import librosa
import torch
from epoch_time import epoch_time
from tqdm import tqdm
from txt_to_stm import txt_to_stm
import pandas as pd
import numpy as np
from padarray import padarray
from to_samples import to_samples
from torch.utils.data import TensorDataset, DataLoader
import audioread
import random
import soundfile as sf
from pathlib import Path

stage='NIST'
sample_rate=8000
window = sample_rate
H=window

transcripts = list(sorted(glob(f'{stage}/*/build/transcription/*.txt')))
len(transcripts)

audio_files=[x.replace('/transcription/', '/audio/').replace('.txt','.wav') for x in transcripts]

for transcript_file in tqdm(transcripts):
    audio_file = transcript_file.replace('/transcription/', '/audio/').replace('.txt','.wav')
    if not os.path.exists(audio_file):
        print('missing', audio_file)
        continue
        
    # Create split dirs
    audio_dir=os.path.dirname(audio_file)
    audio_split_dir=audio_dir.replace('/audio', '/audio_split')
    Path(audio_split_dir).mkdir(parents=True, exist_ok=True)
    transcript_dir=os.path.dirname(transcript_file)
    transcript_split_dir=transcript_dir.replace('/transcription', '/transcription_split')
    Path(transcript_split_dir).mkdir(parents=True, exist_ok=True)
    
    # Load audio
    file = "_".join(os.path.basename(transcript_file).split("_")[:-1])
    channel = os.path.basename(transcript_file).split("_")[-1].split(".")[-2]
    transcript_df = pd.read_csv(transcript_file, sep = "\n", header = None, names = ["content"])
    result = txt_to_stm(transcript_df, file, channel)
    speech=[(float(x[-3]), float(x[-2]), x[-1]) for x in result if len(x)==6]
    x_np,sr=librosa.load(audio_file, sr=sample_rate)
    with audioread.audio_open(audio_file) as f:
        sr = f.samplerate
    if sr != sample_rate:
        print('RESIZING', sr, audio_file)
        sf.write(audio_file, x_np, sample_rate)
        
    # Split audio
    speech_segments=[(int(a*sample_rate), int(b*sample_rate), words) for (a,b,words) in speech]
    for i, (lower, upper, words) in enumerate(speech_segments):
        audio_split_file=f"{audio_file[0:-4].replace('/audio/','/audio_split/')}_{i:03d}.wav"
        sf.write(audio_split_file, x_np[lower:upper], sample_rate)

        transcript_split_file=f"{transcript_file[0:-4].replace('/transcription/','/transcription_split/')}_{i:03d}.txt"
        with open(transcript_split_file,'w') as f:
            f.write(words)





