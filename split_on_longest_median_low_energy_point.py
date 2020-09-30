import numpy as np
from unidecode import unidecode
from itertools import groupby
from operator import itemgetter
import noisereduce as nr
from normalize import normalize
from clip_ends import clip_ends

def split_on_longest_median_low_energy_point(sound, window = 100, threshold = 0.3, min_gap = 10, clip=0.0005):
    audio_moving=np.convolve(np.abs(sound), np.ones((window,))/window, mode='same') 
    amplitudes=np.sort(audio_moving)
    n_amp=audio_moving.shape[0]
    cutoff=amplitudes[int(n_amp*threshold)]
    silence_mask=audio_moving < cutoff
    groups = [[i for i, _ in group] for key, group in groupby(enumerate(silence_mask), key=itemgetter(1)) if key]
    boundaries=[(x[0],x[-1]) for x in groups]
    silences=[(x,y) for x,y in boundaries if y-x > min_gap]
    if not silences:
        return [(sound, 0, sound.shape[0]), None]
    longest_silence=list(sorted([(y-x,(x,y)) for x,y in silences]))[-1][1]
    start,end = longest_silence
    if end-start < window:
        return [(sound, 0, sound.shape[0]), None]
    midpoint = start + (end-start)//2
    left, (startL, endL) = clip_ends(sound[0:midpoint], clip)
    right, (startR, endR) = clip_ends(sound[midpoint:], clip)
    return [(left, startL, endL), (right, midpoint+startR, midpoint+endR)]
