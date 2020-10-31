# Afterburner: Correct ASR pred output to match ASR ground truth output

import pandas as pd
import numpy as np
import os, pickle, random, torch
from tqdm.auto import tqdm
from count_parameters import count_parameters
from initialize_weights import initialize_weights
from calculate_cer import calculate_cer
from calculate_wer import calculate_wer
from prediction_to_string import prediction_to_string
from torchtext.data import Field, BucketIterator, Iterator, TabularDataset
import torch.nn as nn
from seq_to_seq import *

error_correction_training_fn='foo.tsv'
model_fn='save/new_afterburner/afterburner_300.pt'

graphemes=[' ','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q',
           'r','s','t','u','v','w','x','y','à','á','â','ã','è','é','ê','ì','í','ò',
           'ó','ô','õ','ù','ú','ý','ă','đ','ĩ','ũ','ơ','ư','ạ','ả','ấ','ầ','ẩ','ẫ',
           'ậ','ắ','ằ','ẳ','ẵ','ặ','ẹ','ẻ','ẽ','ế','ề','ể','ễ','ệ',
           'ỉ','ị','ọ','ỏ','ố','ồ','ổ','ỗ','ộ','ớ','ờ','ở','ỡ','ợ',
           'ụ','ủ','ứ','ừ','ử','ữ','ự','ỳ','ỷ','ỹ']

MAX_LENGTH=496
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
tokenize=lambda x: [y for y in x]

SRC = Field(tokenize = tokenize, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True, 
            batch_first = True)

TRG = Field(tokenize = tokenize, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True, 
            batch_first = True)

train_data = TabularDataset(
    path=error_correction_training_fn,
    format='tsv',
    fields=[('trg', TRG), ('src', SRC)])

batch_size=32
train_iterator = Iterator(train_data, batch_size=batch_size)

## Model

MIN_FREQ=1

SRC.build_vocab(graphemes, min_freq = MIN_FREQ)

TRG.build_vocab(graphemes, min_freq = MIN_FREQ)
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
INPUT_DIM, OUTPUT_DIM
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

enc = Encoder(INPUT_DIM, 
              HID_DIM, 
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_PF_DIM, 
              ENC_DROPOUT, 
              device,
              MAX_LENGTH)

dec = Decoder(OUTPUT_DIM, 
              HID_DIM, 
              DEC_LAYERS, 
              DEC_HEADS, 
              DEC_PF_DIM, 
              DEC_DROPOUT, 
              device,
              MAX_LENGTH)

SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters')

model.apply(initialize_weights);
if os.path.exists(model_fn):
    model.load_state_dict(torch.load(model_fn))
    print('reloaded trained model')

model.eval();
R=[]
for i, batch in enumerate(tqdm(train_iterator)):
    src = batch.src.to(device)
    trg = batch.trg.to(device)
    output, _ = model(src, trg[:,:-1])
    output_dim = output.shape[-1]
    output = output.contiguous().view(-1, output_dim)
    trg = trg[:,1:].contiguous().view(-1)   
    prediction=prediction_to_string(TRG, batch_size, output, False)
    gold=prediction_to_string(TRG, batch_size, trg, True)   
    for hyp,au in zip(prediction, gold):
        R.append((au,hyp,calculate_cer(hyp, au),calculate_wer(hyp, au)))

R=[(au.strip(), hyp.strip(), cer, wer) for au, hyp, cer, wer in R if '<pad>' not in au]

results=pd.DataFrame(R, columns=['Gold', 'Pred', 'CER', 'WER'])
results['GOLD_n_words']=results['Gold'].apply(lambda x: len(x.split(' ')))
results['GOLD_n_chars']=results['Gold'].apply(lambda x: len(x))
results['CER_pct']=results.CER/results['GOLD_n_chars']
results['WER_pct']=results.CER/results['GOLD_n_words']
results=results[results.Gold != '<pad>']

print('mean WER', results.WER_pct.mean(), 'mean CER', results.CER_pct.mean())
