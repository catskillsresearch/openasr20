#!/usr/bin/env python
# coding: utf-8

# # ASR post-processing corrector pred vs gold

# ## TRAIN

# In[ ]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pylab as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import os, pickle
from multiprocessing import Pool
from Cfg import Cfg
from RecordingCorpus import RecordingCorpus
from tqdm.auto import tqdm
from SplitCorpus import SplitCorpus
from load_pretrained_model import load_pretrained_model
from listen_and_transcribe import listen_and_transcribe


# ## Load training examples

# In[ ]:


if __name__=="__main__":
    language='vietnamese'
    phase='build'
    release='b30'
    C = Cfg('NIST', 16000, language, phase, release)
    with Pool(16) as pool:
        recordings = RecordingCorpus(C, pool)


# In[ ]:


splits=SplitCorpus.transcript_split(C, recordings)


# In[ ]:


max_duration=33
model = load_pretrained_model(C, 0)


# ## For each BUILD subsplit match gold with pred of split audio

# In[ ]:


pairs=[]

for artifact in splits.artifacts:
    gold=artifact.target.value
    audio=artifact.source.value
    transcript=listen_and_transcribe(C, model, max_duration, gold, audio)
    pred=' '.join([z for x,y,z in transcript])
    pairs.append((pred, gold))


# In[ ]:


len([x for x,y in pairs if len(x)==0])


# In[ ]:


pairs = [(x.lower(),y.lower()) for x,y in pairs if len(x)>0]


# In[ ]:


len(pairs)


# ## Save training set

# In[ ]:


training='\n'.join([f'{x.strip()}\t{y.strip()}' for x,y in pairs])


# In[ ]:


error_correction_training_fn='foo.tsv'


# In[ ]:


with open(error_correction_training_fn, 'w', encoding='utf-8') as f:
    f.write(training)


# In[ ]:


training[0:100]


# ## Create consolidated vocabulary

# In[ ]:


graphemes=list(sorted(set([x for x in training if x not in ['\n', '\t']])))


# In[ ]:


graphemes=[' ','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q',
           'r','s','t','u','v','w','x','y','à','á','â','ã','è','é','ê','ì','í','ò',
           'ó','ô','õ','ù','ú','ý','ă','đ','ĩ','ũ','ơ','ư','ạ','ả','ấ','ầ','ẩ','ẫ',
           'ậ','ắ','ằ','ẳ','ẵ','ặ','ẹ','ẻ','ẽ','ế','ề','ể','ễ','ệ',
           'ỉ','ị','ọ','ỏ','ố','ồ','ổ','ỗ','ộ','ớ','ờ','ở','ỡ','ợ',
           'ụ','ủ','ứ','ừ','ử','ữ','ự','ỳ','ỷ','ỹ']


# ## Build model

# In[ ]:


MAX_LENGTH=max([max(len(a), len(b)) for a,b in pairs])+50


# In[ ]:


MAX_LENGTH=496

import random
import torch
from torchtext.data import Field, BucketIterator

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

## Data loader

from torchtext.data import Iterator

from torchtext.data import TabularDataset
train_data = TabularDataset(
    path=error_correction_training_fn,
    format='tsv',
    fields=[('trg', TRG), ('src', SRC)])

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

from seq_to_seq import *

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

from count_parameters import count_parameters

print(f'The model has {count_parameters(model):,} trainable parameters')

import os


from initialize_weights import initialize_weights

model.apply(initialize_weights);
if os.path.exists(model_fn):
    model.load_state_dict(torch.load(model_fn))


# ## Trainer

# In[ ]:


LEARNING_RATE = 0.0005
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)


# In[ ]:


model.train();


# In[ ]:


batch_size=32
train_iterator = Iterator(train_data, batch_size=batch_size)


# In[ ]:


print(f'{len(train_iterator)} batches / epoch')


# In[ ]:


epoch_loss = 9999999999999999
j=0


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt

def progress_bar(ax, progress):
    ax.plot(progress)
    fig.canvas.draw()
    
fig,ax = plt.subplots(1,1)
ax.set_xlabel('epochs')
ax.set_ylabel('Loss')


# In[ ]:


losses = []
while epoch_loss > 0.1:
    epoch_loss = 0
    for i, batch in enumerate(train_iterator):
        src = batch.src.to(device)
        trg = batch.trg.to(device)
        optimizer.zero_grad()
        output, _ = model(src, trg[:,:-1])
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        epoch_loss += loss.item()
    j += 1
    losses.append(epoch_loss)
    progress_bar(ax, losses)


# In[ ]:


get_ipython().system('mkdir -p save/new_afterburner')
model_fn='save/new_afterburner/afterburner_300.pt'
torch.save(model.state_dict(), model_fn)
