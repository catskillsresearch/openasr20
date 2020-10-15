from Cfg import Cfg
from glob import glob
from package_DEV import package_DEV
from load_pretrained_amharic_model import load_pretrained_amharic_model
C = Cfg('NIST', 16000, 'amharic', 'dev', '102') 
model = load_pretrained_amharic_model(C, 1)
files=list(sorted(glob(f'{C.audio_split_dir}/*.wav')))
translations=model.transcribe(paths2audio_files=files, batch_size=16)
package_DEV(C, files, translations)
