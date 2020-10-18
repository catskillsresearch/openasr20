import librosa, os
import numpy as np
import pandas as pd
from glob import glob
from tqdm.auto import tqdm
from ArtifactsVector import ArtifactsVector
from Corpus import Corpus
from plot_log_population import plot_log_population
from plot_log_population2 import plot_log_population2
from sample_statistics import sample_statistics as stats
from text_of_file import text_of_file

class SplitCorpus (Corpus):
    
    def __init__(self, _config, _artifacts):
        super().__init__(_config, _artifacts)
        self.population = ArtifactsVector(_config, _artifacts)

    @classmethod
    def transcript_split(cls, _config, _recordings):
        _artifacts = []
        for artifact in _recordings.artifacts:
            _artifacts.extend(artifact.transcript_split())
        return cls(_config, _artifacts)
    
    @classmethod
    def split_on_silence(cls, _config, _recordings, _goal_length_in_seconds = 33):
        _artifacts = []
        for artifact in tqdm(_recordings.artifacts):
            try:
                _artifacts.extend(artifact.split_on_silence(goal_length_in_seconds=_goal_length_in_seconds))
            except:
                print("BIG PROBLEMO")
                return artifact
        return cls(_config, _artifacts)

    def visualization(self):
        plot_log_population(self.population.N_splits_per_root, 'Splits per 10-minute recording', '# splits per recording', '# recordings with this many splits', 100)
        plot_log_population(self.population.word_lengths_in_graphemes, 'Word lengths', 'Graphemes/word', 'Words with this many graphemes', 12)
        plot_log_population(self.population.samples_per_grapheme,      'Audio samples per grapheme',           'Samples/grapheme', 'Graphemes that are this long in samples', 100)
        plot_log_population(self.population.samples_per_word, 	       'Audio samples per word',               'Samples/word', 'Words that are this long in samples', 100)
        plot_log_population(self.population.split_length_in_words,     'Splits with this many words',          'word length', 'splits', 100)
        plot_log_population(self.population.split_length_in_graphemes, "splits with this many graphemes",      'grapheme length', 'splits', 100)
        plot_log_population(self.population.split_length_in_seconds,   "Splits with this many seconds length", 'sample length (seconds)', 'splits', 100)

    def sample_statistics(self):
        R = [('Words',      '#Words',      'Distinct words in all recordings',         self.population.N_all_words),
             ('Graphemes',  '#Graphemes',  'Distinct graphemes in all transcriptions', self.population.N_all_graphemes),
             ('Splits',     '#Splits',     'Splits in all recordings', 		       self.n_artifacts)]
        R.extend(stats('Words', 'Length in graphemes', self.population.word_lengths_in_graphemes))
        R.extend(self.population.sample_statistics())
        return pd.DataFrame(R, columns=self.columns).sort_values(by=['Corpus', 'Units', 'Measurement']).reset_index(drop=True)

    def diff_visualization(self, new):
        plot_log_population2(self.population.word_lengths_in_graphemes, new.population.word_lengths_in_graphemes,
                             'Word lengths', 'Graphemes/word', 'Words with this many graphemes', 12)
        plot_log_population2(self.population.samples_per_grapheme, new.population.samples_per_grapheme,
                             'Audio samples per grapheme', 'Samples/grapheme', 'Graphemes that are this long in samples', 100)
        plot_log_population2(self.population.samples_per_word, new.population.samples_per_word,
	                     'Audio samples per word', 'Samples/word', 'Words that are this long in samples', 100)
        plot_log_population2(self.population.split_length_in_words, new.population.split_length_in_words,
                             'Splits with this many words', 'word length', 'splits', 100)
        plot_log_population2(self.population.split_length_in_graphemes, new.population.split_length_in_graphemes,
                             "splits with this many graphemes", 'grapheme length', 'splits', 100)
        plot_log_population2(self.population.split_length_in_seconds, new.population.split_length_in_seconds,
                             "Splits with this many seconds length", 'sample length (seconds)', 'splits', 100)

    def diff_sample_statistics(self, new):
        df_old = self.sample_statistics()
        df_new = new.sample_statistics()
        df = pd.merge(df_old, df_new, how='inner', on=df_old.columns.values.tolist()[0:-1])
        return df[df.Value_x != df.Value_y]
    
    def check_vocabulary_change(self, new):
        return list(sorted(set(self.population.all_words).difference(set(new.population.all_words))))
