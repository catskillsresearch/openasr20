import numpy as np
import matplotlib.pylab as plt
from IPython.display import Audio
from Artifact import Artifact
from aggressive_clip_ends import aggressive_clip_ends

class AudioArtifact (Artifact):

    def calc_stats(self):
        self.n_samples = len(self.value)
        self.n_seconds = self.n_samples/self.C.sample_rate
        
    def __init__(self, _config,  _value):
        super().__init__(_config, _value)
        self.calc_stats()

    def aggressively_clip(self):
        self.value = aggressive_clip_ends(self.value, self.C.sample_rate)[0]
        self.calc_stats()
        
    def display(self, transcription = None):
        display(Audio(self.value, rate=self.C.sample_rate))
        plt.figure(figsize=(15,4))
        T = np.arange(self.n_samples)/self.C.sample_rate
        plt.plot(T, self.value);
        plt.xlabel('seconds')
        plt.ylabel('amplitude');
        plt.title(transcription)
        plt.show()
        plt.close()
