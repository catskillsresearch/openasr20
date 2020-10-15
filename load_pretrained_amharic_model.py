from ruamel.yaml import YAML
import nemo.collections.asr as nemo_asr
from reshuffle_samples import reshuffle_samples
from omegaconf import DictConfig
from glob import glob

def load_pretrained_amharic_model(C, device=None):
    models=list(sorted(glob(f'save/nemo_{C.language}/*.ckpt')))
    if len(models):
        last_fn=models[-1]
        model = nemo_asr.models.EncDecCTCModel.load_from_checkpoint(last_fn)
        print('loaded', last_fn)
        return model.cuda(device=device)
