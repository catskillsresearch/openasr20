# coding: utf-8

import logging, json, copy, nemo
from Cfg import Cfg
from ruamel.yaml import YAML
from omegaconf import DictConfig
import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl
from amharic_vocabulary import amharic_vocabulary

logger = logging.getLogger()
logger.setLevel(logging.INFO)
C = Cfg('NIST', 8000, 'amharic') 
config_path = './NeMo/examples/asr/conf/config.yaml'
yaml = YAML(typ='safe')
with open(config_path) as f:
    params = yaml.load(f)
train_manifest=f'{C.build_dir}/train_manifest.json'
test_manifest=f'{C.build_dir}/test_manifest.json'
params['model']['train_ds']['manifest_filepath'] = train_manifest
params['model']['validation_ds']['manifest_filepath'] = test_manifest
quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")
quartznet.change_vocabulary(new_vocabulary=amharic_vocabulary)
quartznet.setup_training_data(train_data_config=params['model']['train_ds'])
quartznet.setup_validation_data(val_data_config=params['model']['validation_ds'])
trainer = pl.Trainer(gpus=[0], max_epochs=1, amp_level='O1', precision=16)
trainer.fit(quartznet)
