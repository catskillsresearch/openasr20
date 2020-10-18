from Sample import Sample
from AudioArtifact import AudioArtifact
from PhraseArtifact import PhraseArtifact

class AudioTextSample(Sample):

    def __init__(self, _config,  _key, _audio, _transcription):
        super().__init__(_key,
                         AudioArtifact(_config, _audio),
                         PhraseArtifact(_config, _transcription))

    def display(self):
        print('KEY', self.key)
        print('SOURCE')
        self.source.display(self.target.romanized)
        print('TARGET')
        self.target.display()
        print()

    def good_split(self):
        if self.target.n_graphemes > 0:
            if self.source.n_seconds <= 4:
                if self.source.n_samples > 370:
                    return True
        return False

