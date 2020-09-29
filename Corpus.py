class Corpus:

    def __init__(self, _config,  _artifacts):
        self.C = _config
        self.artifacts = _artifacts
        self.n_artifacts = len(self.artifacts)
        self.columns = ['Corpus', 'Units', 'Measurement', 'Value']
        self.seconds_of_speech=sum([artifact.source.n_seconds for artifact in _artifacts])
        self.hours_of_speech=self.seconds_of_speech/(60*60)
