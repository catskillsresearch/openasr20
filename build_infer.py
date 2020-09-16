#!/usr/bin/env python
# coding: utf-8

# # Interactive train and test audio samples limited in length audio normalized

import sys
sys.path.append('/home/catskills/Desktop/openasr20/end2end_asr_pytorch')

import os
os.environ['IN_JUPYTER']='True'

from glob import glob
from utils import constant
from utils.data_loader import SpectrogramDataset, AudioDataLoader, BucketingSampler
from utils.functions import load_model
import json, logging, os, sys
import numpy as np

language='amharic'
stage='NIST'

audio_fns=list(sorted(glob('NIST/openasr20_amharic/build/audio_split_normalized_to_18db/BABEL_OP3_307_14229_20140503_233516_inLine_*.wav')))

L=[]
for audio in audio_fns:
    L.append(f'{audio},infer.txt')

manifest_file_path=f'analysis/{language}/size_1.csv'
with open(manifest_file_path,'w') as f:
    f.write('\n'.join(L))

model_dir=f'save/{language}_end2end_asr_pytorch_drop0.1_cnn_batch12_4_vgg_layer4'

args=constant.args
args.continue_from=None
args.cuda = True
args.labels_path = f'analysis/{language}/{language}_characters.json'
args.lr = 1e-4
args.name = f'{language}_end2end_asr_pytorch_drop0.1_cnn_batch12_4_vgg_layer4'
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
args.sample_rate=8000
args.train_manifest_list = [manifest_file_path]
args.continue_from=f'{model_dir}/best_model.th'

args.augment=True

audio_conf = dict(sample_rate=args.sample_rate,
                  window_size=args.window_size,
                  window_stride=args.window_stride,
                  window=args.window,
                  noise_dir=args.noise_dir,
                  noise_prob=args.noise_prob,
                  noise_levels=(args.noise_min, args.noise_max))

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

if constant.args.continue_from:
        model, opt, epoch, metrics, loaded_args, label2id, id2label = load_model(
            constant.args.continue_from)
        start_epoch = epoch  # index starts from zero
        verbose = constant.args.verbose
else:
    model = init_transformer_model(constant.args, label2id, id2label)
    opt = init_optimizer(constant.args, model, "noam")

start_epoch = epoch
metrics = None
loaded_args = None
verbose = True

constant.USE_CUDA=True

train_data = SpectrogramDataset(audio_conf, manifest_filepath_list=args.train_manifest_list, 
                                label2id=label2id, normalize=True, augment=args.augment)

loss_type = args.loss
model = model.cuda(0)

args.batch_size = 1
train_sampler = BucketingSampler(train_data, batch_size=args.batch_size)
train_loader = AudioDataLoader(train_data, num_workers=args.num_workers, batch_sampler=train_sampler)

smoothing = constant.args.label_smoothing

model.eval();

R = []
valid_loader = train_loader
total_valid_loss, total_valid_cer, total_valid_wer, total_valid_char, total_valid_word = 0, 0, 0, 0, 0
for i, (data) in enumerate(valid_loader):
    src, tgt, src_percentages, src_lengths, tgt_lengths = data
    src = src.cuda()
    tgt = tgt.cuda()
    pred, gold, hyp_seq, gold_seq = model(src, src_lengths, tgt, verbose=False)

    seq_length = pred.size(1)

    try:
        strs_gold, strs_hyps = [], []
        for ut_gold in gold_seq:
            str_gold = ""
            for x in ut_gold:
                if int(x) == constant.PAD_TOKEN:
                    break
                str_gold = str_gold + id2label[int(x)]
            strs_gold.append(str_gold)
        for ut_hyp in hyp_seq:
            str_hyp = ""
            for x in ut_hyp:
                if int(x) == constant.PAD_TOKEN:
                    break
                str_hyp = str_hyp + id2label[int(x)]
            strs_hyps.append(str_hyp)
    except Exception as e:
        print(e)
        logging.info("NaN predictions")
        continue

    for j in range(len(strs_hyps)):
        strs_hyps[j] = strs_hyps[j].replace(constant.SOS_CHAR, '').replace(constant.EOS_CHAR, '')
        strs_gold[j] = strs_gold[j].replace(constant.SOS_CHAR, '').replace(constant.EOS_CHAR, '')
        R.append(strs_hyps[j])

with open(f'analysis/{language}/TEST_BUILD.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(R))
