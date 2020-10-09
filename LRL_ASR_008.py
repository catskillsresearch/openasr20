#!/usr/bin/env python
# coding: utf-8

# # Low Resource Language ASR model
# ### Lars Ericson, Catskills Research Company, OpenASR20

# In[ ]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pylab as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ## Configuration

# In[ ]:


from Cfg import Cfg
C = Cfg('NIST', 8000, 'amharic') 


# ## Recording Corpus

# In[ ]:


from RecordingCorpus import RecordingCorpus
from multiprocessing import Pool


# In[ ]:


if __name__ == '__main__':
    with Pool(16) as pool:
        recordings = RecordingCorpus(C, pool)


# In[ ]:


recordings.sample_statistics()


# In[ ]:


recordings.visualization()


# In[ ]:


recordings.artifacts[0].display()


# ## Split corpus

# In[ ]:


from SplitCorpus import SplitCorpus

fat=SplitCorpus.from_recordings(C, recordings)


# In[ ]:


fat.sample_statistics()


# ## Split corpus down to 1 or best

# In[ ]:


from SubSplitCorpus import SubSplitCorpus


# In[ ]:


subsplits=SubSplitCorpus(fat, min_words=2)


# ## Figure out why I'm still seeing a 11.98 second clip and fix that and redo subsplit

# In[ ]:


long_ones=[x for x in subsplits.artifacts if x.source.n_seconds >=  16]
len(long_ones)


# In[ ]:


artifact=long_ones[0]
artifact.display()


# ## Save results

# In[ ]:


len(subsplits.problems)


# In[ ]:


import pickle


# In[ ]:


with open('bfgpu.pkl', 'wb') as f:
    pickle.dump(subsplits,f)


# In[ ]:


get_ipython().system('ls -l bfgpu.pkl')


# ## Check if I'm losing words

# In[ ]:


dp=fat.check_vocabulary_change(subsplits)
len(dp)
print(f'lost {len(dp)} words' if len(dp) else 'no lost words!')


# ## Check I'm not losing graphemes

# In[ ]:


fat_stats = fat.sample_statistics()
sub_stats=subsplits.sample_statistics()

fat_stats[fat_stats.Units=="#Graphemes"].Value==sub_stats[sub_stats.Units=="#Graphemes"].Value


# ## Fix problems in sample statistics

# In[ ]:


import pandas as pd
pd.options.display.float_format = '{:.4f}'.format


# In[ ]:


vars(artifact.source)


# In[ ]:


foo=[x for x in subsplits.artifacts if x.source.n_seconds > 4]
len(foo)


# In[ ]:


370/8000


# In[ ]:


from ArtifactsVector import ArtifactsVector


# In[ ]:


S=subsplits.artifacts
good=[x for x in subsplits.artifacts if x.source.n_samples > 370]
bad=[x for x in subsplits.artifacts if x.source.n_samples <=370]
len(good), len(bad)


# In[ ]:


subsplits.artifacts=good
subsplits.problems=bad
subsplits.population=ArtifactsVector(C, subsplits.artifacts)


# In[ ]:


subsplits.sample_statistics()


# ## Fix problems in visualization

# In[ ]:


subsplits.visualization()


# ## Show change in sample stats and visualizations

# In[ ]:


fat.diff_sample_statistics(subsplits)


# In[ ]:


fat.diff_visualization(subsplits)


# ## Make a test training corpus

# In[ ]:


get_ipython().system('/bin/rm -rf frob')
get_ipython().system('mkdir -p frob/audio')
get_ipython().system('mkdir -p frob/text')


# In[ ]:


import soundfile as sf
clips=[]
for i, artifact in enumerate(tqdm(subsplits.artifacts)):
    sound = artifact.source.value
    fn=f"frob/audio/clip_{i}.wav"
    sf.write(fn, sound, C.sample_rate)
    clips.append(fn)


# In[ ]:


texts=[]
for i, artifact in enumerate(tqdm(subsplits.artifacts)):
    text = artifact.target.value
    fn=f"frob/text/text_{i}.txt"
    with open(fn,'w', encoding='utf-8') as f:
        f.write(text)
    texts.append(fn)


# In[ ]:


manifest_fn="frob/manifest.csv"


# In[ ]:


with open(manifest_fn, 'w') as f:
    f.write('\n'.join([f'{a},{b}' for a,b in zip(clips, texts)]))


# ## ASR end-to-end speech-to-grapheme model stacked on top of grapheme-to-grapheme corrector model

# In[ ]:


C.extension='_gradscaler'
C.batch_size=12
C.save_every = 1
C.start_from = 246


# In[ ]:


import json, sys, os, librosa, random, math, time, torch
sys.path.append('/home/catskills/Desktop/openasr20/end2end_asr_pytorch')
os.environ['IN_JUPYTER']='True'
import numpy as np
import pandas as pd

from itertools import groupby
from operator import itemgetter
import soundfile as sf
from utils import constant
from utils.functions import load_model
from utils.data_loader import SpectrogramDataset, AudioDataLoader, BucketingSampler
from clip_ends import clip_ends
import torch.optim as optim
import torchtext
from torchtext.data import Field, BucketIterator
from torchtext.data import TabularDataset
import matplotlib.ticker as ticker
from IPython.display import Audio
from unidecode import unidecode
from seq_to_seq import *


# In[ ]:


C.analysis_dir = f'analysis/{C.language}'
C.grapheme_dictionary_fn = f'{C.analysis_dir}/{C.language}_characters.json'


# In[ ]:


C.model_name=f'{C.language}_{C.sample_rate}_end2end_asr_pytorch_drop0.1_cnn_batch12_4_vgg_layer4{C.extension}'
C.model_dir=f'save/{C.model_name}'
C.best_model=f'{C.model_dir}/best_model.th'


# In[ ]:


args=constant.args
args.continue_from=None
args.cuda = True
args.labels_path = C.grapheme_dictionary_fn
args.lr = 1e-4
args.name = C.model_name
args.save_folder = f'save'
args.epochs = 1000
args.save_every = 1
args.feat_extractor = f'vgg_cnn'
args.dropout = 0.1
args.num_layers = 4
args.num_heads = 8
args.dim_model = 512
args.dim_key = 64
args.dim_value = 64
args.dim_input = 161
args.dim_inner = 2048
args.dim_emb = 512
args.shuffle=True
args.min_lr = 1e-6
args.k_lr = 1
args.sample_rate=C.sample_rate
args.continue_from=C.best_model
args.augment=True
audio_conf = dict(sample_rate=args.sample_rate,
                  window_size=args.window_size,
                  window_stride=args.window_stride,
                  window=args.window,
                  noise_dir=args.noise_dir,
                  noise_prob=args.noise_prob,
                  noise_levels=(args.noise_min, args.noise_max))


# In[ ]:


with open(args.labels_path, 'r') as label_file:
    labels = str(''.join(json.load(label_file)))
# add PAD_CHAR, SOS_CHAR, EOS_CHAR
labels = constant.PAD_CHAR + constant.SOS_CHAR + constant.EOS_CHAR + labels
label2id, id2label = {}, {}
count = 0
for i in range(len(labels)):
    if labels[i] not in label2id:
        label2id[labels[i]] = count
        id2label[count] = labels[i]
        count += 1
    else:
        print("multiple label: ", labels[i])

model, opt, epoch, metrics, loaded_args, label2id, id2label = load_model(constant.args.continue_from)


# In[ ]:


constant.USE_CUDA=True


# In[ ]:


start_epoch = 0
loaded_args = None
loss_type = args.loss

if constant.USE_CUDA:
    model = model.cuda(0)


# In[ ]:


num_epochs = constant.args.epochs


# In[ ]:


num_epochs = 1


# In[ ]:


import logging
logging.info(model)


# In[ ]:


args.batch_size=8


# ## Gradually skim the cream

# In[ ]:


from TrainerVanilla import TrainerVanilla


# In[ ]:


from grab_text import grab_text


# In[ ]:


threshold = 0.2


# In[ ]:


for zzz in range(100):
    manifest_fn="cream.csv"
    args.train_manifest_list=[manifest_fn]
    train_data = SpectrogramDataset(audio_conf, manifest_filepath_list=args.train_manifest_list, label2id=label2id, normalize=True, augment=args.augment)
    train_sampler = BucketingSampler(train_data, batch_size=args.batch_size)
    train_loader = AudioDataLoader(train_data, num_workers=args.num_workers, batch_sampler=train_sampler)
    trainer = TrainerVanilla()
    sample_results = trainer.train(model, train_loader, train_sampler, opt, loss_type, 
                                   start_epoch, num_epochs, label2id, id2label, just_once=True)
    n_samples=len(sample_results)
    print('n_samples', n_samples)

    performance=pd.DataFrame(sample_results, columns=['Pred', 'Gold', 'CER', 'WER'])
    performance['GOLD_n_chars']=performance['Gold'].str.len()
    performance['CER_pct']=performance['CER']/performance['GOLD_n_chars']
    performance['GOLD_n_words']=performance['Gold'].apply(lambda x: len(x.split(' ')))
    performance['WER_pct']=performance['WER']/performance['GOLD_n_words']
    performance.CER_pct.hist(bins=100);
    performance.plot(kind='scatter', x='GOLD_n_words', y='CER_pct');
    threshold=max(0.1, threshold*0.95)
    print("threshold", threshold)
    dft=performance[performance.CER_pct < threshold]
    n_threshold=len(dft)
    display(dft.head())
    print('n_threshold',n_threshold)
    print('n_threshold/n_samples', n_threshold/n_samples)
    gold=performance[performance.CER_pct < threshold].Gold.values
    with open(manifest_fn, 'r') as f:
        manifest=f.readlines()
    M1=[[y.strip() for y in x.split(',')] for x in manifest]
    M2={grab_text(txtfn):(wavfn,txtfn) for wavfn, txtfn in M1}
    cream=[M2[x] for x in gold]
    with open('cream.csv', 'w') as f:
        for a,b in cream:
            f.write(f'{a},{b}\n')


# In[ ]:


with pd.option_context("display.max_rows", 1000): 
    display(performance.sort_values("GOLD_n_chars", ascending=False).head(32))


# ## Afterburner training to correct piece-by-piece quasi-phonemic ("learned phonemic") translations

# ### Gather training data from a rerun on all subsplits

# In[ ]:


manifest_fn="frob/manifest.csv"
args.train_manifest_list=[manifest_fn]
train_data = SpectrogramDataset(audio_conf, manifest_filepath_list=args.train_manifest_list, label2id=label2id, normalize=True, augment=args.augment)
train_sampler = BucketingSampler(train_data, batch_size=args.batch_size)
train_loader = AudioDataLoader(train_data, num_workers=args.num_workers, batch_sampler=train_sampler)
trainer = TrainerVanilla()
sample_results = trainer.train(model, train_loader, train_sampler, opt, loss_type, 
                               start_epoch, num_epochs, label2id, id2label, just_once=True)
n_samples=len(sample_results)
print('n_samples', n_samples)

import pickle
with open('sample_results.pkl', 'wb') as f:
    pickle.dump(sample_results, f)
# ### Format training data TSV file

# In[ ]:


error_correction_training_fn='error_correction.tsv'

training='\n'.join([f'{gold.strip()}\t{pred.strip()}' for pred,gold,z,w in sample_results])with open(error_correction_training_fn, 'w', encoding='utf-8') as f:
    f.write(training)
# In[ ]:


with open(error_correction_training_fn, 'r', encoding='utf-8') as f:
    training=f.read()
GP=training.split('\n')


# In[ ]:


graphemes=list(sorted(set([x for x in training if x not in ['\n', '\t']])))


# In[ ]:


len(graphemes)


# In[ ]:


GP=[x.split('\t') for x in GP]


# In[ ]:


MAX_LENGTH=max([max(len(a), len(b)) for a,b in GP])


# In[ ]:


MAX_LENGTH


# In[ ]:


model


# In[ ]:


get_ipython().system('ls error_correction.tsv')


# In[ ]:


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


# In[ ]:


from torchtext.data import Iterator


# In[ ]:


from torchtext.data import TabularDataset
train_data = TabularDataset(
    path=error_correction_training_fn,
    format='tsv',
    fields=[('trg', TRG), ('src', SRC)])


# In[ ]:


batch_size=128


# In[ ]:


train_iterator = Iterator(train_data, batch_size=batch_size)


# In[ ]:


MIN_FREQ=1


# In[ ]:


SRC.build_vocab(graphemes, min_freq = MIN_FREQ)


# In[ ]:


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


# In[ ]:


from seq_to_seq import *


# In[ ]:


enc = Encoder(INPUT_DIM, 
              HID_DIM, 
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_PF_DIM, 
              ENC_DROPOUT, 
              device,
              MAX_LENGTH)


# In[ ]:


dec = Decoder(OUTPUT_DIM, 
              HID_DIM, 
              DEC_LAYERS, 
              DEC_HEADS, 
              DEC_PF_DIM, 
              DEC_DROPOUT, 
              device,
              MAX_LENGTH)


# In[ ]:


SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]


# In[ ]:


model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)


# In[ ]:


model_fn='tut6-model.pt'

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')


# In[ ]:


os.path.exists(model_fn)


# In[ ]:


import os

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

model.apply(initialize_weights);
if os.path.exists(model_fn):
    model.load_state_dict(torch.load(model_fn))

LEARNING_RATE = 0.0005
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)


# In[ ]:


model.train();


# In[ ]:


for j in range(1):
    epoch_loss = 0
    print(f"# batches: {len(train_iterator)}")
    for i, batch in enumerate(train_iterator):

        print(f"start step {j} {i} src max {batch.src.max().item} trg max {batch.trg.max().item()}")
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
        print(f'[{i}] {epoch_loss}')
        
    print(j, epoch_loss)


# In[ ]:





# In[ ]:


pred=output.argmax(1).cpu().detach().numpy()
''.join([SRC.vocab.itos[x] for x in src.cpu().numpy()[0]])
silver=''.join([TRG.vocab.itos[x] for x in trg.cpu().numpy()]).split('<eos>')[0]
pred=''.join([TRG.vocab.itos[x] for x in pred]).split('<eos>')[0]
from utils.metrics import calculate_cer, calculate_wer
calculate_cer(pred, silver)
calculate_wer(pred, silver)


# ### Translate BUILD a whole split at a time, using SubSplit splitting method, and correct that

# ### Translate BUILD a whole recording at a time, using SubSplit splitting method, and correct that

# In[ ]:




