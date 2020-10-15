from Cfg import Cfg
from ruamel.yaml import YAML
import nemo.collections.asr as nemo_asr
from reshuffle_samples import reshuffle_samples
from omegaconf import DictConfig

def load_pretrained_amharic_model():
    C = Cfg('NIST', 16000, 'amharic') 
    config_path = 'amharic_16000.yaml'
    yaml = YAML(typ='safe')
    with open(config_path) as f:
        params = yaml.load(f)
    params['model']['train_ds']['manifest_filepath'] = f'{C.build_dir}/train_manifest.json'
    params['model']['validation_ds']['manifest_filepath'] = f'{C.build_dir}/test_manifest.json'
    params['model']['optim']['lr'] = 0.001
    model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")
    model.change_vocabulary(new_vocabulary=params['labels'])
    model.setup_optimization(optim_config=DictConfig(params['model']['optim']))
    model.setup_training_data(train_data_config=params['model']['train_ds'])
    model.setup_validation_data(val_data_config=params['model']['validation_ds'])
    model.load_from_checkpoint('save/nemo_amharic/amharic_20201014_170626_320956_epoch=87.ckpt')
    return C, model.cuda(), params
