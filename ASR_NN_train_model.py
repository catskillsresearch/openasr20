from utils import constant
from utils.functions import load_model
from utils.data_loader import SpectrogramDataset, AudioDataLoader, BucketingSampler
from seq_to_seq import *
from TrainerVanilla import TrainerVanilla
from grab_text import grab_text
sys.path.append('/home/catskills/Desktop/openasr20/end2end_asr_pytorch')

def ASR_NN_train_model():
    import json, sys, os, librosa, random, math, time, torch, logging
    import numpy as np
    import pandas as pd
    from itertools import groupby
    from operator import itemgetter
    import soundfile as sf
    from clip_ends import clip_ends
    import torch.optim as optim
    import torchtext
    from torchtext.data import Field, BucketIterator
    from torchtext.data import TabularDataset
    import matplotlib.ticker as ticker
    from IPython.display import Audio
    from unidecode import unidecode
    from Cfg import Cfg


    C = Cfg('NIST', 8000, 'amharic') 

    num_epochs = constant.args.epochs
    num_epochs = 1

    logging.info(model)
    args.batch_size=8

    threshold = 0.2

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
