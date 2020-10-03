import numpy as np
from unidecode import unidecode
from itertools import groupby
from operator import itemgetter
import noisereduce as nr
from normalize import normalize
from clip_ends import clip_ends
from longest_silence import longest_silence

def split_on_longest_median_low_energy_point(sound, window = 100, threshold = 0.3, min_gap = 10):
    LS, clip = longest_silence(sound, window, threshold, min_gap, clip=0.0005)
    if not LS:
        return [(sound, 0, sound.shape[0]), None]
    start,end = LS
    if end-start < window:
        return [(sound, 0, sound.shape[0]), None]
    midpoint = start + (end-start)//2
    left, (startL, endL) = clip_ends(sound[0:midpoint], clip)
    right, (startR, endR) = clip_ends(sound[midpoint:], clip)
    return [(left, startL, endL), (right, midpoint+startR, midpoint+endR)]
