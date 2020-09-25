import os
from glob import glob
from Corpus import Corpus
from plot_log_population import plot_log_population
from sample_statistics import sample_statistics as stats
import librosa
from tqdm import tqdm
from tokenizer import tokenizer
import numpy as np
import pandas as pd
from text_of_file import text_of_file
from AudioTranscriptionSample import AudioTranscriptionSample

class SplitCorpus (Corpus):

    def __init__(self, _config, recordings):
        self.split_from = recordings
        self.build_roots=[os.path.basename(x)[0:-4] for x in self.split_from.build_transcription_filenames]
        self.build_transcription_split_dir=f'{_config.build_dir}/transcription_split'
        self.build_transcription_split_filenames=glob(f'{self.build_transcription_split_dir}/*.txt')
        self.build_split_targets = [text_of_file(x) for x in tqdm(self.build_transcription_split_filenames)]
        self.build_split_roots = [os.path.basename(x)[0:-4] for x in self.build_transcription_split_filenames]
        self.build_audio_split_nr_dir=f'{_config.build_dir}/audio_split_{_config.sample_rate}'
        self.build_audio_split_nr_filenames=[f'{self.build_audio_split_nr_dir}/{x}.wav' for x in self.build_split_roots]
        self.build_split_sources = [librosa.load(src_fn, sr=_config.sample_rate)[0] for src_fn in tqdm(self.build_audio_split_nr_filenames)]
        artifacts = [AudioTranscriptionSample(afn, audio, tfn, transcription)
                     for afn, audio, tfn, transcription
                     in zip(self.build_audio_split_nr_filenames,
                            self.build_split_sources,
                            self.build_transcription_split_filenames,
                            self.build_split_targets)]
        super().__init__(_config, artifacts)
        self.N_splits_per_root=[len([x for x in self.build_split_roots if x.startswith(y)]) for y in self.build_roots]
        self.N_all_splits=len(self.build_split_roots)
        self.build_split_targets_tokenized=[tokenizer(x) for x in self.build_split_targets]
        self.split_length_in_words=[len(x) for x in self.build_split_targets_tokenized]
        self.split_length_in_samples=np.array([len(x) for x in self.build_split_sources])
        self.split_length_in_seconds=self.split_length_in_samples/_config.sample_rate
        self.split_length_in_graphemes=[len(x) for x in self.build_split_targets]
        self.all_words=list(sorted(set([item for sublist in self.build_split_targets_tokenized for item in sublist])))
        self.N_all_words=len(self.all_words)
        self.all_graphemes=list(sorted(set(''.join(self.all_words))))
        self.N_all_graphemes=len(self.all_graphemes)
        self.word_lengths_in_graphemes=[len(x) for x in self.all_words]
        self.samples_per_word=self.split_length_in_samples/self.split_length_in_words
        self.seconds_per_word=self.samples_per_word/_config.sample_rate
        self.samples_per_grapheme = self.split_length_in_samples/self.split_length_in_graphemes
        self.seconds_per_grapheme = self.samples_per_grapheme/_config.sample_rate


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
             ('Splits',     '#Splits',     'Splits in all recordings', 		   self.N_all_splits)]
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
