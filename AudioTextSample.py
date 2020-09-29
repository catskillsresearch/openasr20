from AudioArtifact import AudioArtifact
from Sample import Sample
from PhraseArtifact import PhraseArtifact
from aggressive_clip_ends import aggressive_clip_ends
from epoch_time import epoch_time
from glob import glob
from matplotlib.pylab import *
from padarray import padarray
from pathlib import Path
from to_samples import to_samples
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from txt_to_stm import txt_to_stm
import audioread, librosa, os, random
import numpy as np

class AudioTextSample(Sample):

    def __init__(self, _config, _root, _key, _audio, _transcription):
        super().__init__(_root, _key,
                         AudioArtifact(_config, _audio),
                         PhraseArtifact(_config, _transcription))

    def aggressive_clip_ends(self):
        audio, bounds = aggressive_clip_ends(self.source.value, self.source.C.sample_rate)
        return AudioTextSample(self.source.C, self, bounds, audio, self.target.value)

    def display(self):
        print('KEY', self.key)
        print('SOURCE')
        self.source.display(self.target.romanized)
        print('TARGET')
        self.target.display()
        print()
