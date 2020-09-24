from glob import glob
import os, librosa
from tqdm import tqdm
from tokenizer import tokenizer
import numpy as np
import pandas as pd
from text_of_file import text_of_file
from plot_log_population import plot_log_population

class Language:

    def __init__(self, language):
        self.stage='NIST'
        self.sample_rate=8000
        self.language=language
        self.load_training_corpus()

    def load_training_corpus(self):
        self.data_dir=f'{self.stage}/openasr20_{self.language}'
        self.build_dir=f'{self.data_dir}/build'
        self.build_transcription_split_dir=f'{self.build_dir}/transcription_split'
        self.build_transcription_split_filenames=glob(f'{self.build_transcription_split_dir}/*.txt')
        self.build_split_targets = [text_of_file(x) for x in tqdm(self.build_transcription_split_filenames)]
        self.build_split_roots = [os.path.basename(x)[0:-4] for x in self.build_transcription_split_filenames]
        self.build_audio_split_nr_dir=f'{self.build_dir}/audio_split_{self.sample_rate}'
        self.build_audio_split_nr_filenames=[f'{self.build_audio_split_nr_dir}/{x}.wav' for x in self.build_split_roots]
        self.build_split_sources = [librosa.load(src_fn, sr=self.sample_rate)[0] for src_fn in tqdm(self.build_audio_split_nr_filenames)]
        self.build_examples = list(zip(self.build_split_roots, self.build_split_sources, self.build_split_targets))
        self.build_transcription_dir=f'{self.build_dir}/transcription'
        self.build_transcription_filenames=glob(f'{self.build_transcription_dir}/*.txt')
        self.N_recordings=len(self.build_transcription_filenames)
        self.build_roots=[os.path.basename(x)[0:-4] for x in self.build_transcription_filenames]
        self.N_splits_per_root=[len([x for x in self.build_split_roots if x.startswith(y)]) for y in self.build_roots]
        self.N_all_splits=len(self.build_split_roots)
        self.build_split_targets_tokenized=[tokenizer(x) for x in self.build_split_targets]
        self.split_length_in_words=[len(x) for x in self.build_split_targets_tokenized]
        self.split_length_in_samples=np.array([len(x) for x in self.build_split_sources])
        self.split_length_in_seconds=self.split_length_in_samples/self.sample_rate
        self.split_length_in_graphemes=[len(x) for x in self.build_split_targets]
        self.all_words=list(sorted(set([item for sublist in self.build_split_targets_tokenized for item in sublist])))
        self.N_all_words=len(self.all_words)
        self.all_graphemes=list(sorted(set(''.join(self.all_words))))
        self.N_all_graphemes=len(self.all_graphemes)
        self.hours_of_build_speech=10*self.N_recordings/60
        self.word_lengths_in_graphemes=[len(x) for x in self.all_words]
        self.samples_per_word=self.split_length_in_samples/self.split_length_in_words
        self.seconds_per_word=self.samples_per_word/self.sample_rate
        self.samples_per_grapheme = self.split_length_in_samples/self.split_length_in_graphemes
        self.seconds_per_grapheme = self.samples_per_grapheme/self.sample_rate

    def statistics(self, corpus, measurement, samples):
        return [(corpus, measurement, 'Mean', np.mean(samples)),
                (corpus, measurement, 'Median', np.median(samples)),
                (corpus, measurement, 'Min', np.min(samples)),
                (corpus, measurement, 'Max', np.max(samples))]
        
    def language_report_visualization(self):
        plot_log_population(self.samples_per_grapheme, 'Audio samples per grapheme', 'Samples/grapheme', 'Graphemes that are this long in samples', 100)
        plot_log_population(self.N_splits_per_root, 'Splits per 10-minute recording', '# splits per recording', '# recordings with this many splits', 100)
        plot_log_population(self.split_length_in_words, 'Splits with this many words', 'word length', 'splits', 100)
        plot_log_population(self.split_length_in_graphemes, "splits with this many graphemes", 'grapheme length', 'splits', 100)
        plot_log_population(self.split_length_in_seconds, "Splits with this many seconds length", 'sample length (seconds)', 'splits', 100)
        plot_log_population(self.word_lengths_in_graphemes, 'Word lengths', 'Graphemes/word', 'Words with this many graphemes', 12)
        plot_log_population(self.split_length_in_seconds, 'Split length in seconds', 'Seconds/split', 'Samples that are this long in seconds', 100)
        plot_log_population(self.split_length_in_words, 'Split length in words', 'Words/split', 'Samples that are this long in words', 100)
        plot_log_population(self.split_length_in_graphemes, 'Split length in graphemes', 'Graphemes/split', 'Samples that are this long in graphemes', 100)
        plot_log_population(self.samples_per_word, 'Audio samples per word', 'Samples/word', 'Words that are this long in samples', 100)
        
    def language_report_statistics(self):
        C = ['Corpus', 'Units', 'Measurement', 'Value']
        R = [('Recordings', '#Recordings', '10-minute training recordings',    	   self.N_recordings),
             ('Recordings', '#Hours',      'Hours of training speech',         	   self.hours_of_build_speech),
             ('Words',      '#Words',      'Distinct words in all recordings', 	   self.N_all_words),
             ('Graphemes',  '#Graphemes',  'Distinct graphemes in all transcriptions', self.N_all_graphemes),
             ('Splits',     '#Splits',     'Splits in all recordings', 		   self.N_all_splits)]
        R.extend(self.statistics('Words', 'Word length in graphemes', self.word_lengths_in_graphemes))
        R.extend(self.statistics('Split Speech', 'Length in samples', self.split_length_in_samples))
        R.extend(self.statistics('Split Speech', 'Length in seconds', self.split_length_in_seconds))
        R.extend(self.statistics('Split Transcription', 'Length in words', self.split_length_in_words))
        R.extend(self.statistics('Split Transcription', 'Length in graphemes', self.split_length_in_graphemes))
        R.extend(self.statistics('Words', 'Length in samples', self.samples_per_word))
        R.extend(self.statistics('Words', 'Length in seconds', self.seconds_per_word))
        R.extend(self.statistics('Graphemes', 'Length in samples', self.samples_per_grapheme))
        R.extend(self.statistics('Graphemes', 'Length in seconds', self.seconds_per_grapheme))
        return pd.DataFrame(R, columns=C).sort_values(by=['Corpus', 'Units', 'Measurement']).reset_index(drop=True)
        
    def language_report(self):
        self.language_report_visualization()
        return self.language_report_statistics()
