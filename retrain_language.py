#!/usr/bin/env python
# coding: utf-8

from Cfg import Cfg
from load_pretrained_model import load_pretrained_model
from reshuffle_samples import reshuffle_samples
from ruamel.yaml import YAML
from omegaconf import DictConfig
import pytorch_lightning as pl
import os, datetime
from ModelCheckpointAtEpochEnd import ModelCheckpointAtEpochEnd

def retrain_language(C, train_fraction=0.8, max_duration = 12.0):
    reshuffle_samples(C, train_fraction)
    model = load_pretrained_model(C, 0)
    config_path = f'{C.language}_{C.sample_rate}.yaml'
    yaml = YAML(typ='safe')
    with open(config_path) as f:
        params = yaml.load(f)
    train_manifest=f'{C.build_dir}/train_manifest.json'
    test_manifest=f'{C.build_dir}/test_manifest.json'
    params['model']['train_ds']['manifest_filepath'] = train_manifest
    pid=os.getpid()
    dt=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_callback = ModelCheckpointAtEpochEnd(
        filepath=C.model_save_dir+f'/{C.language}_'+f'{dt}_{pid}'+'_{epoch:02d}',
        verbose=True,
        save_top_k=-1,
        save_weights_only=False,
        period=1)
    trainer = pl.Trainer(gpus=[0], max_epochs=1000, amp_level='O1', precision=16, checkpoint_callback=checkpoint_callback)
    model.set_trainer(trainer)
    model.setup_training_data(train_data_config=params['model']['train_ds'])

    if train_fraction < 1:
        params['model']['validation_ds']['manifest_filepath'] = test_manifest
        model.setup_validation_data(val_data_config=params['model']['validation_ds'])

    model.setup_optimization(optim_config=DictConfig(params['model']['optim']))
    trainer.fit(model)

if __name__=="__main__":
    import sys
    language=sys.argv[1]
    C = Cfg('NIST', 16000, language, 'build') 
    retrain_language(C, 0.95, 12.0)
