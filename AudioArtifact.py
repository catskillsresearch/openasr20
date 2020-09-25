from Artifact import Artifact

class AudioArtifact (Artifact):

    def __init__(self, _config,  _filename, _value):
        super().__init__(_config, _filename, _value)
        if _value is None:
            return
        self.n_samples = _value.shape[0]
        self.n_seconds = self.n_samples/_config.sample_rate

