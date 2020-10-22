#!/usr/bin/env python
# coding: utf-8

from Cfg import Cfg
from RecordingCorpus import RecordingCorpus
from load_pretrained_model import load_pretrained_model
from multiprocessing import Pool
from transcription import transcription
import sys, pickle
import warnings
warnings.filterwarnings("ignore")

language=sys.argv[1]
version=sys.argv[2]
gpu=int(sys.argv[3])
max_duration=6
print("language", language, "version", version, "gpu", gpu, "max_duration", max_duration)

# Load language model
C = Cfg('NIST', 16000, language, 'dev', version)
model = load_pretrained_model(C, gpu)
if not model:
    print("ERROR: no model")
    quit()

# ## Move onto DEV to visualize, test and refine splitter
if __name__ == '__main__':   
    with Pool(16) as pool:
        recordings = RecordingCorpus(C, pool)

translations = []
for artifact in recordings.artifacts:
    key = artifact.key
    print(key)
    (lng,tfn)=key
    audio = artifact.source.value
    transcript=transcription(C, model, audio, max_duration)
    translations.append((key, transcript))
    save_fn=f'transcription_{lng}_{tfn}.pkl'
    with open(save_fn, 'wb') as f:
        pickle.dump(transcript, f)
    print('saved', save_fn)

fn = f'translation_{language}_{version}_{gpu}_{max_duration}.pkl'

package =  {'fn': fn,
            'language': language,
            'version': version,
            'gpu': gpu,
            'max_duration': max_duration,
            'translations': translations}

with open(fn, 'wb') as f:
    pickle.dump(translations, f)
print('saved', fn)
