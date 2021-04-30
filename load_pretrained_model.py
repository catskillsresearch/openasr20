from ruamel.yaml import YAML
import nemo.collections.asr as nemo_asr
from reshuffle_samples import reshuffle_samples
from omegaconf import DictConfig
from glob import glob

def load_pretrained_model(C, device=None):
    search_dir=f'save/nemo_{C.language}/*.ckpt'
    print("searching", search_dir)
    models=list(sorted(glob(search_dir)))
    if len(models):
        last_fn=models[-1]
        model = nemo_asr.models.EncDecCTCModel.load_from_checkpoint(last_fn)
        print('loaded', last_fn)
        return model.cuda(device=device)
    else:
        print("NO MODELS IN", search_dir)
        quit()

if __name__=="__main__":
    from Cfg import Cfg
    C = Cfg('NIST', 16000, 'pashto', 'dev', '001')
    gpu=1
    model = load_pretrained_model(C, gpu)
    if not model:
        print("ERROR: no model")
