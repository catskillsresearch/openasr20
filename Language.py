import pandas as pd
from Cfg import Cfg
from RecordingCorpus import RecordingCorpus
from SplitCorpus import SplitCorpus

class Language:

    def __init__(self, _language):
        self.C = Cfg('NIST', 8000, _language) 
        self.recordings = RecordingCorpus(self.C)
        self.splits = SplitCorpus.from_split_directory(self.C)

    def visualization(self):
        self.recordings.visualization()
        self.splits.visualization()

    def sample_statistics(self):
        df1 = self.recordings.sample_statistics()
        df2 = self.splits.sample_statistics()
        return pd.concat([df1, df2])

    def language_report(self):
        self.visualization()
        return self.sample_statistics()
