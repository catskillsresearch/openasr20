from glob import glob
import librosa
import numpy as np
from runtrainer import runtrainer

def train_language(C):

    os.system("/bin/rm -rf runs")
    files=glob(f'{C.nr_dir}/*')
    n_samples=len(files)
    samples=list(sorted([(librosa.load(audio_file, sr=C.sample_rate)[0].shape[0], audio_file) for audio_file in files]))

    sieve = 500
    sample_ends=(n_samples*np.arange(sieve)/sieve).astype(int).tolist()[1:]+[n_samples]

    # Hand tune when it breaks
    batch_size = {i: C.batch_size for i in range(len(sample_ends))}

    for i, end in enumerate(sample_ends):
        bs = batch_size[i]
        print("------------------------------------------------")
        print(f"[{i}] batch_size {bs} samples {end}")
        runtrainer(C, [y for x,y in samples[0:end]], max(2,10-i), batch_size[i])
