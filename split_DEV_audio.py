# coding: utf-8

import os
from glob import glob
from audio_to_speech_segments import audio_to_speech_segments
from tqdm import tqdm
from silence_detector import silence_detector

def split_DEV_audio(C):
    device, model = silence_detector(C.sample_rate)
    audio_files = list(sorted(glob(f'{C.stage}/openasr20_{C.language}/dev/audio/*.wav')))
    for audio_file in tqdm(audio_files):
        audio_to_speech_segments(sample_rate, device, model, audio_file)
