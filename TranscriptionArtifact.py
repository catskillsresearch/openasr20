from Artifact import Artifact
from tokenizer import tokenizer
from unidecode import unidecode

class TranscriptionArtifact (Artifact):

    def __init__(self, _config, _filename, _value):
        super().__init__(_config, _filename, _value)
        if _value is None:
            return
        self.tokens = tokenizer(_value)
        self.n_words = len(self.tokens)
        self.n_graphemes = len(_value)
        self.romanized = unidecode(self.value)

    def display(self):
        print(self.value, '::', self.romanized)
