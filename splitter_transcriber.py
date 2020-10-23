#!/usr/bin/env python
# coding: utf-8

from Cfg import Cfg
from RecordingCorpus import RecordingCorpus
from load_pretrained_model import load_pretrained_model
from multiprocessing import Pool
from transcription import transcription
import sys, pickle, os, warnings
warnings.filterwarnings("ignore")

def splitter_transcriber(language, release, gpu, max_duration, phase):
    tdir= f'transcriptions/{language}/{phase}/{release}'
    os.system(f'mkdir -p {tdir}')
    print("language", language, "release", release, "gpu", gpu, "max_duration", max_duration)
    C = Cfg('NIST', 16000, language, 'dev', release)
    model = load_pretrained_model(C, gpu)
    if not model:
        print("ERROR: no model")
        quit()
    if __name__ == '__main__':   
        with Pool(16) as pool:
            recordings = RecordingCorpus(C, pool)
    translations = []
    for artifact in recordings.artifacts:
        key = artifact.key
        (lng,tfn)=key
        print("key", key)
        save_fn=f'{tdir}/transcription_{lng}_{tfn}.pkl'
        if os.path.exists(save_fn):
            print("finished", key)
            continue
        print(key)
        audio = artifact.source.value
        transcript=transcription(C, model, audio, max_duration)
        translations.append((key, transcript))
        with open(save_fn, 'wb') as f:
            pickle.dump(transcript, f)
        print('saved', save_fn)

if __name__=="__main__":
    import logging
    logging.getLogger().setLevel(logging.NOTSET)
    language=sys.argv[1]
    release=sys.argv[2]
    gpu=int(sys.argv[3])
    phase=sys.argv[4]
    max_duration=16.5
    splitter_transcriber(language, release, gpu, max_duration, phase)
