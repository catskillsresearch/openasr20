import numpy as np
from sample_statistics import sample_statistics as stats

class ArtifactsVector:

    def __init__(self, _config,  _artifacts):
        self.config = _config
        # Source recording stats
        split_from_roots={x.key[0:-1]:0 for x in _artifacts}
        for artifact in _artifacts:
            split_from_roots[artifact.key[0:-1]] += 1
        self.N_splits_per_root=[y for x,y in split_from_roots.items()]
        # Audio Source
        self.split_length_in_samples=np.array([artifact.source.n_samples for artifact in _artifacts])
        self.split_length_in_seconds=np.array([artifact.source.n_seconds for artifact in _artifacts])
        # Text Target
        self.split_words=[artifact.target.tokens for artifact in _artifacts]
        self.split_length_in_words=np.array([artifact.target.n_words for artifact in _artifacts])
        self.split_length_in_graphemes=np.array([artifact.target.n_graphemes for artifact in _artifacts])
        # Text in Audio
        self.samples_per_word=self.split_length_in_samples/self.split_length_in_words
        self.seconds_per_word=self.samples_per_word/self.config.sample_rate
        self.samples_per_grapheme = self.split_length_in_samples/self.split_length_in_graphemes
        self.seconds_per_grapheme = self.samples_per_grapheme/self.config.sample_rate
        # Words
        self.all_words=list(sorted(set([item for sublist in self.split_words for item in sublist])))
        self.N_all_words=len(self.all_words)
        # Graphemes
        self.all_graphemes=list(sorted(set(''.join(self.all_words))))
        self.N_all_graphemes=len(self.all_graphemes)
        # Graphemes in Words
        self.word_lengths_in_graphemes=[len(x) for x in self.all_words]

    def sample_statistics(self):
        R = []
        R.extend(stats('Split Speech', 'Length in samples', self.split_length_in_samples))
        R.extend(stats('Split Speech', 'Length in seconds', self.split_length_in_seconds))
        R.extend(stats('Split Transcription', 'Length in words', self.split_length_in_words))
        R.extend(stats('Split Transcription', 'Length in graphemes', self.split_length_in_graphemes))
        R.extend(stats('Words', 'Length in samples', self.samples_per_word))
        R.extend(stats('Words', 'Length in seconds', self.seconds_per_word))
        R.extend(stats('Graphemes', 'Length in samples', self.samples_per_grapheme))
        R.extend(stats('Graphemes', 'Length in seconds', self.seconds_per_grapheme))
        return R
