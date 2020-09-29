from AudioArtifact import AudioArtifact

class RecordingArtifact(AudioArtifact):

    def __init__(self, _config,  _value, _filename):
        super().__init__(_config, _value)
        self.filename = _filename
