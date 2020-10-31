import os, pickle
from multiprocessing import Pool
from Cfg import Cfg
from load_pretrained_model import load_pretrained_model
from RecordingCorpus import RecordingCorpus
from listen_and_transcribe import listen_and_transcribe
from tqdm.auto import tqdm

def translator(language, phase, release, max_duration):
    C = Cfg('NIST', 16000, language, phase, release)
    model = load_pretrained_model(C, 0)
    with Pool(16) as pool:
        recordings = RecordingCorpus(C, pool)
    tdir= f'pred/{language}/{phase}/{release}'
    os.system(f'mkdir -p {tdir}')
    translations = []
    for artifact in recordings.artifacts:
        key = artifact.key
        (lng,tfn)=key
        print("key", key)
        save_fn=f'{tdir}/transcription_{lng}_{tfn}.pkl'
        if os.path.exists(save_fn):
            print("finished", key)
            continue
        audio = artifact.source.value
        gold=artifact.gold()
        transcript=listen_and_transcribe(C, model, max_duration, gold, audio)
        translations.append((key, transcript))
        with open(save_fn, 'wb') as f:
            pickle.dump(transcript, f)
        print('saved', save_fn)
        print()

if __name__=="__main__":
    import sys
    language, phase, release, max_duration = sys.argv[1:]
    max_duration=float(max_duration)
    translator(language, phase, release, max_duration)
