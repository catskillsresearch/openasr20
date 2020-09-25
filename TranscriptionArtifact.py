from Artifact import Artifact
from tokenizer import tokenizer

class TranscriptionArtifact (Artifact):

    def __init__(self, _config, _filename, _value):
        super().__init__(_config, _filename, _value)
        if _value is None:
            return
        self.tokens = tokenizer(_value)
        self.n_words = len(self.tokens)
        self.n_graphemes = len(_value)
