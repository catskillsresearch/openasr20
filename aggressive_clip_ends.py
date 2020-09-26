import numpy as np
from itertools import groupby
from operator import itemgetter
import noisereduce as nr
from normalize import normalize
from clip_ends import clip_ends

def aggressive_clip_ends(audio, sample_rate):
    window = 2048
    if audio.shape[0] < window:
        return normalize(audio)
    N=10
    cutoff=np.max(audio[0:20])
    silence_mask=np.abs(audio) < cutoff
    groups = [[i for i, _ in group] for key, group in groupby(enumerate(silence_mask), key=itemgetter(1)) if key]
    boundaries=[(x[0],x[-1]) for x in groups]
    silences=[(x/sample_rate,(y-N)/sample_rate) for x,y in boundaries if y-x > window]
    if not len(silences):
        return normalize(audio)
    longest_noise=sorted([(y-x,(x,y)) for x,y in silences])[-1][1]
    noisy_segment=tuple(int(x*sample_rate) for x in longest_noise)
    noisy_part = audio[noisy_segment[0]:min(noisy_segment[0]+window, noisy_segment[1])]
    reduced_noise = nr.reduce_noise(audio_clip=audio, noise_clip=noisy_part, verbose=False)
    normed_reduced_noise=normalize(reduced_noise)
    cutoff=np.max(normed_reduced_noise[0:20])
    return clip_ends(normed_reduced_noise, clip=cutoff)
