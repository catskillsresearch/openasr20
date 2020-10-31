import json
from Cfg import Cfg
from afterburner_model import afterburner_model
import random,  torch
from torchtext.data import Field, Iterator, TabularDataset

def afterburner_pretrained_model(language, phase, release, model_fn):
    C = Cfg('NIST', 16000, language, phase, release)
    error_correction_training_fn=f'traindata_{C.language}.tsv'
    vocabulary_fn=f'vocabulary_{C.language}.csv'
    max_length_fn=f'maxlength_{C.language}.txt'
    with open(vocabulary_fn, 'r', encoding='utf-8') as f:
        graphemes = json.load(f)
    with open(max_length_fn, 'r') as f:
        MAX_LENGTH = int(f.read())
    model, SRC, TRG, device = afterburner_model(graphemes, MAX_LENGTH, model_fn)
    train_data = TabularDataset(
        path=error_correction_training_fn,
        format='tsv',
        fields=[('src', SRC), ('trg', TRG)])
    train_iterator = Iterator(train_data, batch_size=batch_size)
    return C, model, SRC, TRG, device, train_iterator
