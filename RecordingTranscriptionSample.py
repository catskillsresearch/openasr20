import os
import numpy as np
from Sample import Sample
from RecordingArtifact import RecordingArtifact
from TranscriptArtifact import TranscriptArtifact
from AudioTextSample import AudioTextSample
from load_and_resample_if_necessary import load_and_resample_if_necessary
from power_split import power_split

class RecordingTranscriptionSample(Sample):

    def __init__(self, _config, _afn, _tfn):
        _key = (os.path.basename(_afn)[0:-4],)
        _audio = load_and_resample_if_necessary(_config, _afn)
        super().__init__((_config.language,)+_key,
                         RecordingArtifact(_config, _audio, _afn),
                         TranscriptArtifact(_config, _tfn))

    def display(self):
        print('KEY', self.key)
        print('SOURCE')
        self.source.display('10-MINUTE RECORDING')
        print('TARGET')
        self.target.display()
        print()

    def transcript_split(self):
        x_np = self.source.value
        C = self.source.C
        speech=[(float(x[-3]), float(x[-2]), x[-1]) for x in self.target.value if len(x)==6]
        speech_segments=[(int(a*C.sample_rate), int(b*C.sample_rate), words)
                         for (a,b,words) in speech
                         if 'IGNORE' not in words]
        return [AudioTextSample(C, self.key+((lower,upper),), x_np[lower:upper], words)
                for i, (lower, upper, words) in enumerate(speech_segments)]

    def split_on_silence(self, goal_length_in_seconds = 33):
        C = self.source.C
        audio = self.source.value
        cuts, self.gap, T = power_split(C, audio)
        boundaries=np.hstack([[0],cuts,[audio.shape[0]]])
        segments=np.array([(boundaries[i], boundaries[i+1]) for i in range(boundaries.shape[0]-1)])
        return [AudioTextSample(C, self.key+((start,end)), audio[start:end], '  ')
                for start,end in segments]
