#!/usr/bin/env python
# coding: utf-8

import os
from Cfg import Cfg
from RecordingCorpus import RecordingCorpus
from SplitCorpus import SplitCorpus
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")
import soundfile as sf
from tqdm.auto import tqdm

def power_split_DEV(recordings, language, max_duration):
    splits=SplitCorpus.split_on_silence(C, recordings, max_duration)
    os.system(f'rm {C.audio_split_dir}/*.wav')
    for artifact in tqdm(splits.artifacts):
        language, root, start, end = artifact.key
        filename = f'{C.audio_split_dir}/{root}_{start}_{end}.wav'
        audio = artifact.source.value
        sf.write(filename, audio, C.sample_rate)

if __name__=="__main__":
    import sys
    language=sys.argv[1]
    max_duration=float(sys.argv[2])
    C = Cfg('NIST', 16000, language, 'dev')
    with Pool(16) as pool:
        recordings = RecordingCorpus(C, pool)
    power_split_DEV(recordings, language, max_duration)
