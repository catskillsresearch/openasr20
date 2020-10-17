# Usage: python build_run.py 115 1 8

from Cfg import Cfg
from glob import glob
from package_DEV import package_DEV
from load_pretrained_amharic_model import load_pretrained_amharic_model
import sys
version=sys.argv[1]
gpu=int(sys.argv[2])
batch_size=int(sys.argv[3])
print("version", version, "gpu", gpu)
C = Cfg('NIST', 16000, 'amharic', 'build', version)
model = load_pretrained_amharic_model(C, gpu)
files=list(sorted(glob(f'{C.audio_split_dir}/*.wav')))
translations=model.transcribe(paths2audio_files=files, batch_size=batch_size)
package_DEV(C, files, translations)
