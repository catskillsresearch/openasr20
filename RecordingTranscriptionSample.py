import os
from Sample import Sample
from RecordingArtifact import RecordingArtifact
from TranscriptArtifact import TranscriptArtifact
from AudioTextSample import AudioTextSample
from load_and_resample_if_necessary import load_and_resample_if_necessary
from optimal_split import optimal_split
from split_on_silence import split_on_silence as sos

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

    def split_on_silence(self, t_lower=0.0001, t_upper=0.8, window = 500, min_gap = 10, goal_length_in_seconds = 3):
        C = self.source.C
        sample_rate = C.sample_rate
        audio = self.source.value
        A=sos(audio, t_lower, t_upper, window, min_gap, sample_rate, goal_length_in_seconds)
        return [AudioTextSample(C, self.key+(region,), audio, '  ')
                for (audio, region) in A]
