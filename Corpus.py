class Corpus:

    def __init__(self, _config,  _artifacts):
        self.config = _config
        self.artifacts = _artifacts
        self.n_artifacts = len(self.artifacts)
        self.columns = ['Corpus', 'Units', 'Measurement', 'Value']
