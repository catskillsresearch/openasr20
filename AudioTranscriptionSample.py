from Sample import Sample
from AudioArtifact import AudioArtifact
from TranscriptionArtifact import TranscriptionArtifact

class AudioTranscriptionSample(Sample):

    def __init__(self, _afn, _audio, _tfn, _transcription):
        super().__init__(AudioArtifact(_afn, _audio), TranscriptionArtifact(_tfn, _transcription))
