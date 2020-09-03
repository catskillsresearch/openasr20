#!/usr/bin/env python
# coding: utf-8

from glob import glob
from audio_to_speech_segments import audio_to_speech_segments
from tqdm import tqdm
from silence_detector import silence_detector

stage='NIST'

sample_rate=8000
device, model = silence_detector(sample_rate)
audio_files = list(sorted(glob(f'{stage}/*/dev/audio/*.wav')))
for audio_file in tqdm(audio_files):
    audio_to_speech_segments(sample_rate, device, model, audio_file)

