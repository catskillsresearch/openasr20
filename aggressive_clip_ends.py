import numpy as np
from itertools import groupby
from operator import itemgetter
import noisereduce as nr
from normalize import normalize
from clip_ends import clip_ends
from longest_silence import longest_silence

def aggressive_clip_ends(audio, sample_rate):
    noisy_segment, cutoff = longest_silence(audio)
    if not noisy_segment:
        return normalize(audio), (0, len(audio))
    noisy_part = audio[noisy_segment[0]:noisy_segment[1]]
    reduced_noise = nr.reduce_noise(audio_clip=audio, noise_clip=noisy_part, verbose=False)
    normed_reduced_noise=normalize(reduced_noise)
    return clip_ends(normed_reduced_noise, clip=cutoff)
