# coding: utf-8

import os
import pandas as pd
from glob import glob
from itertools import groupby
from operator import itemgetter
import noisereduce as nr
from normalize import normalize

def split_audio_clip_by_silence(gold_src, sample_rate):
    N=10
    audio_moving=np.convolve(gold_src**2, np.ones((N,))/N, mode='same') 
    amplitudes=np.sort(audio_moving)
    n_amp=audio_moving.shape[0]
    cutoff=amplitudes[int(n_amp*0.15)]
    silence_mask=audio_moving < cutoff
    groups = [[i for i, _ in group] for key, group in groupby(enumerate(silence_mask), key=itemgetter(1)) if key]
    boundaries=[(x[0],x[-1]) for x in groups]
    silences=[(x/sample_rate,(y-N)/sample_rate) for x,y in boundaries if y-x > 0.1*sample_rate]
    longest_noise=sorted([(y-x,(x,y)) for x,y in silences])[-1][1]
    noisy_segment=tuple(int(x*sample_rate) for x in longest_noise)
    noisy_part = audio[noisy_segment[0]:noisy_segment[1]]
    reduced_noise = nr.reduce_noise(audio_clip=audio, noise_clip=noisy_part, verbose=True)
    normed_reduced_noise=normalize(reduced_noise)
    return silences, normed_reduced_noise
