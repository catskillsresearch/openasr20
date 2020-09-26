import numpy as np
import matplotlib.pylab as plt
from IPython.display import Audio
from Artifact import Artifact

class AudioArtifact (Artifact):

    def __init__(self, _config,  _filename, _value):
        super().__init__(_config, _filename, _value)
        if _value is None:
            return
        self.n_samples = _value.shape[0]
        self.n_seconds = self.n_samples/_config.sample_rate

    def display(self):
        display(Audio(self.value, rate=self.C.sample_rate))
        plt.figure(figsize=(15,4))
        T = np.arange(self.n_samples)/self.C.sample_rate
        plt.plot(T, self.value);
        plt.xlabel('seconds')
        plt.ylabel('amplitude');
        plt.title(self.filename)
