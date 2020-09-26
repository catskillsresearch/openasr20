import librosa, os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from ArtifactsVector import ArtifactsVector
from AudioTranscriptionSample import AudioTranscriptionSample
from Corpus import Corpus
from plot_log_population import plot_log_population
from sample_statistics import sample_statistics as stats
from text_of_file import text_of_file

class SplitCorpus (Corpus):

    def __init__(self, _config, _recordings):
        self.build_transcription_dir=f'{_config.build_dir}/transcription_split'
        build_transcription_filenames=glob(f'{self.build_transcription_dir}/*.txt')
        build_roots = [os.path.basename(x)[0:-4] for x in build_transcription_filenames]

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
        self.population = ArtifactsVector(_config, self.artifacts)

    def visualization(self):
        plot_log_population(self.population.N_splits_per_root,         'Splits per 10-minute recording',       '# splits per recording', '# recordings with this many splits', 100)
        plot_log_population(self.population.word_lengths_in_graphemes, 'Word lengths',                         'Graphemes/word', 'Words with this many graphemes', 12)
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
