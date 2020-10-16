from Cfg import Cfg
from glob import glob
from package_DEV import package_DEV
from load_pretrained_amharic_model import load_pretrained_amharic_model
import sys
version=sys.argv[1]
gpu=int(sys.argv[2])
print("version", version, "gpu", gpu)
C = Cfg('NIST', 16000, 'amharic', 'dev', version)
model = load_pretrained_amharic_model(C, gpu)
files=list(sorted(glob(f'{C.audio_split_dir}/*.wav')))
translations=model.transcribe(paths2audio_files=files, batch_size=16)
if 0:
    for file, translation in list(zip(files, translations))[0:20]:
        print(file, translation)
package_DEV(C, files, translations)
