from glob import glob
import tqdm
from rts import rts
from Corpus import Corpus
import pandas as pd

class RecordingCorpus (Corpus):

    def __init__(self, _config, _pool):
        build_transcription_filenames=glob(f'{_config.build_dir}/transcription/*.txt')
        build_audio_filenames=[fn.replace('/transcription/', '/audio/').replace('.txt','.wav') for fn in build_transcription_filenames]
        tasks = [(_config,x,y) for x,y in zip(build_audio_filenames, build_transcription_filenames)]
        artifacts = [a for a in tqdm.tqdm(_pool.imap(rts, tasks), total=len(tasks)) if a is not None]
        super().__init__(_config, artifacts)

    def visualization(self):
        return

    def sample_statistics(self):
        R = [('Recordings', '#Recordings', '10-minute recordings',    	   self.n_artifacts),
             ('Recordings', '#Hours',      'Hours of training speech',     self.hours_of_speech)]
        return pd.DataFrame(R, columns=self.columns).sort_values(by=['Corpus', 'Units', 'Measurement']).reset_index(drop=True)
