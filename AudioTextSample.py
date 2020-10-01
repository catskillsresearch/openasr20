from Sample import Sample
from AudioArtifact import AudioArtifact
from PhraseArtifact import PhraseArtifact

class AudioTextSample(Sample):

    def __init__(self, _config, _root, _key, _audio, _transcription):
        super().__init__(_root, _key,
                         AudioArtifact(_config, _audio),
                         PhraseArtifact(_config, _transcription))

    def aggressively_clip(self):
        source.aggressively_clip()

    def display(self):
        print('KEY', self.key)
        print('SOURCE')
        self.source.display(self.target.romanized)
        print('TARGET')
        self.target.display()
        print()
