from Cfg import Cfg
from glob import glob
from package_DEV import package_DEV
from load_pretrained_model import load_pretrained_model
import sys
language=sys.argv[1]
version=sys.argv[2]
gpu=int(sys.argv[3])
batch_size=int(sys.argv[4])

print("language", language, "version", version, "gpu", gpu, "batch_size", batch_size)
C = Cfg('NIST', 16000, language, 'dev', version)
model = load_pretrained_model(C, gpu)
if not model:
    print("ERROR: no model")
    quit()
files=list(sorted(glob(f'{C.audio_split_dir}/*.wav')))
translations=model.transcribe(paths2audio_files=files, batch_size=batch_size)
package_DEV(C, files, translations)
