# coding: utf-8
# ASR post-processing corrector pred vs gold: TRAINING DATA CREATION

import json
from load_training_examples import load_training_examples
from load_pretrained_model import load_pretrained_model
from listen_and_transcribe import listen_and_transcribe

def afterburner_gold(language, phase, release):
    """For each BUILD subsplit match gold with pred of split audio"""
    C, splits = load_training_examples(language, phase, release)
    max_duration=33
    model = load_pretrained_model(C, 0)
    pairs=[]
    for artifact in splits.artifacts:
        gold=artifact.target.value
        audio=artifact.source.value
        transcript=listen_and_transcribe(C, model, max_duration, gold, audio)
        pred=' '.join([z for x,y,z in transcript])
        pairs.append((pred, gold))
    pairs = [(x.lower(),y.lower()) for x,y in pairs if len(x)>0]
    augment = [(y,y) for x,y in pairs]
    training='\n'.join([f'{x.strip()}\t{y.strip()}' for x,y in pairs+augment])
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
    language='vietnamese'
    phase='build'
    release='b30'
    afterburner_gold(language, phase, release)
