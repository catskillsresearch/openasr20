import numpy as np
from normalize import normalize
from split_on_longest_median_low_energy_point import split_on_longest_median_low_energy_point
from tqdm.auto import tqdm

def good_large_split_unstripped(sound, t_lower, t_upper, window, min_gap, sample_rate, goal_length_in_seconds):
    if goal_length_in_seconds >= sound.shape[0]/sample_rate:
        return [(0,sound.shape[0])]
    audio_normalized=normalize(sound)
    Right = None
    for t_poke in np.linspace(t_lower, t_upper, 10):
        [(left, startL, endL), Right]=split_on_longest_median_low_energy_point(audio_normalized, window, t_poke, min_gap)
        if Right is not None:
            break
    if Right is None:
        return [(startL, endL)]
    (right, startR, endR)=Right
    t_left = (endL-startL)/sample_rate
    t_right = (endR-startR)/sample_rate
    L = good_large_split_unstripped(left, t_lower, t_upper, window, min_gap, sample_rate, goal_length_in_seconds)
    R = good_large_split_unstripped(right, t_lower, t_upper, window, min_gap, sample_rate, goal_length_in_seconds)
    endL = L[-1][-1]
    R = [(x+endL,y+endL) for x,y in R]
    return L+R
