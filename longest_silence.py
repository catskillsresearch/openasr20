import numpy as np
from itertools import groupby
from operator import itemgetter

def longest_silence(sound, window = 100, threshold = 0.3, min_gap = 10, clip=0.0005):
    audio_moving=np.convolve(np.abs(sound), np.ones((window,))/window, mode='same') 
    amplitudes=np.sort(audio_moving)
    n_amp=audio_moving.shape[0]
    cutoff=amplitudes[int(n_amp*threshold)]
    silence_mask=audio_moving < cutoff
    groups = [[i for i, _ in group] for key, group in groupby(enumerate(silence_mask), key=itemgetter(1)) if key]
    boundaries=[(x[0],x[-1]) for x in groups]
    silences=[(x,y) for x,y in boundaries if y-x > min_gap]
    if not silences:
        return None, cutoff
    else:
        return (list(sorted([(y-x,(x,y)) for x,y in silences]))[-1][1], cutoff)
