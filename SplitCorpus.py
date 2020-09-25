import os
from glob import glob
from Corpus import Corpus
from plot_log_population import plot_log_population
from sample_statistics import sample_statistics as stats
import librosa
from tqdm import tqdm

import numpy as np
import pandas as pd
from text_of_file import text_of_file
from AudioTranscriptionSample import AudioTranscriptionSample

class SplitCorpus (Corpus):

    def __init__(self, _config, _recordings):
        split_from_roots=[x.key for x in _recordings.artifacts]
        self.build_transcription_dir=f'{_config.build_dir}/transcription_split'
        build_transcription_filenames=glob(f'{self.build_transcription_dir}/*.txt')
        build_roots = [os.path.basename(x)[0:-4] for x in build_transcription_filenames]
        self.N_splits_per_root=[len([x for x in build_roots if x.startswith(y)]) for y in split_from_roots]
        build_targets = [text_of_file(x) for x in tqdm(build_transcription_filenames)]
        self.build_audio_nr_dir=f'{_config.build_dir}/audio_split_{_config.sample_rate}'
        build_audio_nr_filenames=[f'{self.build_audio_nr_dir}/{x}.wav' for x in build_roots]
        build_sources = [librosa.load(src_fn, sr=_config.sample_rate)[0] for src_fn in tqdm(build_audio_nr_filenames)]
        artifacts = [AudioTranscriptionSample(_config, key.split('Line_')[0]+'Line', key, afn, audio, tfn, transcription)
                     for key, afn, audio, tfn, transcription
                     in zip(build_roots,
                            build_audio_nr_filenames,
                            build_sources,
                            build_transcription_filenames,
                            build_targets)]
        super().__init__(_config, artifacts)
        self.analysis()
        
    def analysis(self):
        # Population
        self.build_targets_tokenized=[artifact.target.tokens for artifact in self.artifacts]
        self.split_length_in_words=np.array([artifact.target.n_words for artifact in self.artifacts])
        self.split_length_in_graphemes=np.array([artifact.target.n_graphemes for artifact in self.artifacts])
        self.split_length_in_samples=np.array([artifact.source.n_samples for artifact in self.artifacts])
        self.split_length_in_seconds=np.array([artifact.source.n_seconds for artifact in self.artifacts])

        # Dictionary
        self.all_words=list(sorted(set([item for sublist in self.build_targets_tokenized for item in sublist])))
        self.N_all_words=len(self.all_words)

        # Writing system
        self.all_graphemes=list(sorted(set(''.join(self.all_words))))
        self.N_all_graphemes=len(self.all_graphemes)

        # Statistics
        self.word_lengths_in_graphemes=[len(x) for x in self.all_words]
        self.samples_per_word=self.split_length_in_samples/self.split_length_in_words
        self.seconds_per_word=self.samples_per_word/self.config.sample_rate
        self.samples_per_grapheme = self.split_length_in_samples/self.split_length_in_graphemes
        self.seconds_per_grapheme = self.samples_per_grapheme/self.config.sample_rate

    def visualization(self):
        plot_log_population(self.N_splits_per_root,         'Splits per 10-minute recording',       '# splits per recording', '# recordings with this many splits', 100)
        plot_log_population(self.word_lengths_in_graphemes, 'Word lengths',                         'Graphemes/word', 'Words with this many graphemes', 12)
        plot_log_population(self.samples_per_grapheme,      'Audio samples per grapheme',           'Samples/grapheme', 'Graphemes that are this long in samples', 100)        
        plot_log_population(self.samples_per_word,          'Audio samples per word',               'Samples/word', 'Words that are this long in samples', 100)
        plot_log_population(self.split_length_in_words,     'Splits with this many words',          'word length', 'splits', 100)
        plot_log_population(self.split_length_in_graphemes, "splits with this many graphemes",      'grapheme length', 'splits', 100)
        plot_log_population(self.split_length_in_seconds,   "Splits with this many seconds length", 'sample length (seconds)', 'splits', 100)

    def sample_statistics(self):
        R = [('Words',      '#Words',      'Distinct words in all recordings', 	   self.N_all_words),
             ('Graphemes',  '#Graphemes',  'Distinct graphemes in all transcriptions', self.N_all_graphemes),
             ('Splits',     '#Splits',     'Splits in all recordings', 		   self.n_artifacts)]
        R.extend(stats('Words', 'Length in graphemes', self.word_lengths_in_graphemes))
        R.extend(stats('Split Speech', 'Length in samples', self.split_length_in_samples))
        R.extend(stats('Split Speech', 'Length in seconds', self.split_length_in_seconds))
        R.extend(stats('Split Transcription', 'Length in words', self.split_length_in_words))
        R.extend(stats('Split Transcription', 'Length in graphemes', self.split_length_in_graphemes))
        R.extend(stats('Words', 'Length in samples', self.samples_per_word))
        R.extend(stats('Words', 'Length in seconds', self.seconds_per_word))
        R.extend(stats('Graphemes', 'Length in samples', self.samples_per_grapheme))
        R.extend(stats('Graphemes', 'Length in seconds', self.seconds_per_grapheme))
        return pd.DataFrame(R, columns=self.columns).sort_values(by=['Corpus', 'Units', 'Measurement']).reset_index(drop=True)
