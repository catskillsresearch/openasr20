from Artifact import Artifact
from tokenizer import tokenizer
from unidecode import unidecode

class PhraseArtifact (Artifact):

    def __init__(self, _config, _value):
        super().__init__(_config, _value)
        self.tokens = tokenizer(_value)
        self.value = ' '.join(self.tokens)
        self.n_words = len(self.tokens)
        self.n_graphemes = len(self.value)
        self.romanized = unidecode(self.value)

    def display(self):
        print(self.value, '::', self.romanized)
