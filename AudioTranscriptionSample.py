from Sample import Sample
from AudioArtifact import AudioArtifact
from TranscriptionArtifact import TranscriptionArtifact
from aggressive_clip_ends import aggressive_clip_ends

class AudioTranscriptionSample(Sample):

    def __init__(self, _config, _root, _key,  _afn, _audio, _tfn, _transcription):
        super().__init__(_root, _key,
                         AudioArtifact(_config, _afn, _audio),
                         TranscriptionArtifact(_config, _tfn, _transcription))

    def aggressive_clip_ends(self):
        audio = aggressive_clip_ends(self.source.value, self.source.C.sample_rate)
        return AudioTranscriptionSample(self.source.C, self.root, self.key,
                                        self.source.filename, audio,
                                        self.target.filename, self.target.value)

    def display(self):
        print('KEY', self.key)
        print('SOURCE')
        self.source.display(self.target.romanized)
        print('TARGET')
        self.target.display()
        print()
