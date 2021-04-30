# coding: utf-8
# ASR post-processing corrector pred vs gold: TRAINING DATA CREATION

import librosa, json
from tqdm.auto import tqdm
from json_lines_load import json_lines_load
from load_training_examples import load_training_examples
from load_pretrained_model import load_pretrained_model
from transcribe import transcribe
import logging
logging.getLogger('nemo_logger').setLevel(logging.ERROR)

def afterburner_gold(C, max_duration):
    """For each BUILD subsplit match gold with pred of split audio"""
    manifests=json_lines_load(f'{C.build_dir}/all_manifest.json')
    model = load_pretrained_model(C, 0)
    pairs=[]
    # "audio_filepath", "duration", "text"
    for manifest in tqdm(manifests):
        gold=manifest['text']
        audio,sr1=librosa.load(manifest['audio_filepath'], sr=C.sample_rate)
        pred=transcribe(C, model, audio)
        pairs.append((pred, gold))
    pairs = [(x.lower(),y.lower()) for x,y in pairs if len(x)>0]
    augment = [(y,y) for x,y in pairs]
    tdata = list(set(pairs+augment))
    training='\n'.join([f'{x.strip()}\t{y.strip()}' for x,y in tdata])
    error_correction_training_fn=f'traindata_{C.language}.tsv'

    # Save training set
    with open(error_correction_training_fn, 'w', encoding='utf-8') as f:
        f.write(training)
        print('saved', error_correction_training_fn)

    # Save vocabulary
    graphemes=list(sorted(set([x for x in training if x not in ['\n', '\t']])))
    vocabulary_fn=f'vocabulary_{C.language}.csv'
    with open(vocabulary_fn, 'w', encoding='utf-8') as f:
        json.dump(graphemes, f)
        print('saved', vocabulary_fn)

    # Save maximum input/output length
    MAX_LENGTH=max([max(len(a), len(b)) for a,b in pairs])+50
    max_length_fn=f'maxlength_{C.language}.txt'
    with open(max_length_fn, 'w') as f:
        f.write(str(MAX_LENGTH))
        print('saved', max_length_fn)

if __name__=="__main__":
    from Cfg import Cfg
    language='somali'
    phase='build'
    release='000'
    max_duration=10
    C = Cfg('NIST', 16000, language, phase, release) 
    afterburner_gold(C, max_duration)
