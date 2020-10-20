#!/usr/bin/env python
# coding: utf-8

import os, tqdm
from Cfg import Cfg
from RecordingCorpus import RecordingCorpus
from SplitCorpus import SplitCorpus
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")
import soundfile as sf

def pool_power_split_DEV(pool, C, max_duration):
    os.system(f'rm -rf {C.audio_split_dir}')
    os.system(f'mkdir {C.audio_split_dir}')
    recordings = RecordingCorpus(C, pool)
    splits=SplitCorpus.pool_split_on_silence(C, pool, recordings, max_duration)
    too_short = C.sample_rate * 0.2
    for artifact in tqdm.tqdm(splits.artifacts):
        language, root, start, end = artifact.key
        filename = f'{C.audio_split_dir}/{root}_{start}_{end}.wav'
        audio = artifact.source.value
        if audio.shape[0] >= too_short:
            sf.write(filename, audio, C.sample_rate)

if __name__=="__main__":
    import sys
    language=sys.argv[1]
    max_duration=float(sys.argv[2])
    C = Cfg('NIST', 16000, language, 'dev')
    with Pool(16) as pool:
        pool_power_split_DEV(pool, C, max_duration)
