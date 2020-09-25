import os
from glob import glob
from Corpus import Corpus
import numpy as np
import pandas as pd
from AudioTranscriptionSample import AudioTranscriptionSample

class RecordingCorpus (Corpus):

    def __init__(self, _config):
        self.build_transcription_dir=f'{_config.build_dir}/transcription'
        self.build_transcription_filenames=glob(f'{self.build_transcription_dir}/*.txt')
        self.build_audio_dir=f'{_config.build_dir}/audio'
        self.build_audio_filenames=[fn.replace('/audio/', '/transcription/').replace('.txt','.wav') for fn in self.build_transcription_filenames]
        artifacts = [AudioTranscriptionSample(_config, os.path.basename(afn)[0:-4], afn, None, tfn, None)
                     for afn, tfn in zip(self.build_audio_filenames, self.build_transcription_filenames)]
        super().__init__(_config, artifacts)
        self.hours_of_build_speech=10*self.n_artifacts/60

    def visualization(self):
        pass

    def sample_statistics(self):
        R = [('Recordings', '#Recordings', '10-minute training recordings',    	   self.n_artifacts),
             ('Recordings', '#Hours',      'Hours of training speech',         	   self.hours_of_build_speech)]
        return pd.DataFrame(R, columns=self.columns).sort_values(by=['Corpus', 'Units', 'Measurement']).reset_index(drop=True)

