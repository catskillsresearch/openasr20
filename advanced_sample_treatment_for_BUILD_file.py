import os
import librosa
import numpy as np
from normalize import normalize
from clip_ends import clip_ends
from itertools import groupby
from operator import itemgetter
import noisereduce as nr
from normalize import normalize
import soundfile as sf
import warnings

def advanced_sample_treatment_for_BUILD_file(C, src_fn):
    warnings.filterwarnings("ignore")
    tgt_fn=src_fn.replace('/audio_split/',f'/audio_split_{C.sample_rate}/')
    fsplit=os.path.basename(src_fn)[0:-4].split('_')
    start,end=fsplit[-2:]
    start,end=(float(start), float(end))
    gold_src,sr=librosa.load(src_fn, sr=C.sample_rate)
    (N, threshold, min_gap)=(100, 0.4, 0.1)
    audio_moving=np.convolve(gold_src**2, np.ones((N,))/N, mode='same') 
    amplitudes=np.sort(audio_moving)
    n_amp=audio_moving.shape[0]
    cutoff=amplitudes[int(n_amp*threshold)]
    silence_mask=audio_moving < cutoff
    groups = [[i for i, _ in group] for key, group in groupby(enumerate(silence_mask), key=itemgetter(1)) if key]
    boundaries=[(x[0],x[-1]-N) for x in groups]
    boundaries=[(x,y) for x,y in boundaries if y-x > N]
    try:
        longest_noise=sorted([(y-x,(x,y)) for x,y in boundaries])[-1][1]
    except:
        longest_noise=None
    if longest_noise:
        noisy_segment=tuple(int(x) for x in longest_noise)
        noisy_part = gold_src[noisy_segment[0]:noisy_segment[1]]
        reduced_noise = nr.reduce_noise(audio_clip=gold_src, noise_clip=noisy_part, verbose=False)
    else:
        reduced_noise=gold_src
    normed_reduced_noise=normalize(reduced_noise)
    tgt=clip_ends(normed_reduced_noise, 0.0008)
    sf.write(tgt_fn, tgt, C.sample_rate)
