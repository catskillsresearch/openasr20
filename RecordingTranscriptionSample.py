import os
from Sample import Sample
from RecordingArtifact import RecordingArtifact
from TranscriptArtifact import TranscriptArtifact
from AudioTextSample import AudioTextSample
from load_and_resample_if_necessary import load_and_resample_if_necessary

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

    def split(self):
        x_np = self.source.value
        C = self.source.C
        speech=[(float(x[-3]), float(x[-2]), x[-1]) for x in self.target.value if len(x)==6]
        speech_segments=[(int(a*C.sample_rate), int(b*C.sample_rate), words)
                         for (a,b,words) in speech
                         if 'IGNORE' not in words]
        return [AudioTextSample(C, self.key+((lower,upper),), x_np[lower:upper], words)
                for i, (lower, upper, words) in enumerate(speech_segments)]
