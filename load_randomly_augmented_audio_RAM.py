from torch.utils.data import Dataset
import numpy as np
import os
import subprocess
import torchaudio
from augment_audio_with_sox_RAM import augment_audio_with_sox_RAM

def load_randomly_augmented_audio_RAM(audio, sample_rate=16000, tempo_range=(0.85, 1.15), gain_range=(-6, 8)):
    """
    Picks tempo and gain uniformly, applies it to the utterance by using sox utility.
    Returns the augmented utterance.
    """
    low_tempo, high_tempo = tempo_range
    tempo_value = np.random.uniform(low=low_tempo, high=high_tempo)
    low_gain, high_gain = gain_range
    gain_value = np.random.uniform(low=low_gain, high=high_gain)
    return augment_audio_with_sox_RAM(audio, sample_rate, tempo_value, gain_value)
