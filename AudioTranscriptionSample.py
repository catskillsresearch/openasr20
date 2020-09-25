from Sample import Sample
from AudioArtifact import AudioArtifact
from TranscriptionArtifact import TranscriptionArtifact

class AudioTranscriptionSample(Sample):

    def __init__(self, _config, _root, _key,  _afn, _audio, _tfn, _transcription):
        super().__init__(_root, _key,
                         AudioArtifact(_config, _afn, _audio),
                         TranscriptionArtifact(_config, _tfn, _transcription))
