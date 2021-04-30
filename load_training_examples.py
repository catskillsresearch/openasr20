from multiprocessing import Pool
from Cfg import Cfg
from RecordingCorpus import RecordingCorpus
from SplitCorpus import SplitCorpus

def load_training_examples(language, phase, release):
    C = Cfg('NIST', 16000, language, phase, release)
    with Pool(16) as pool:
        recordings = RecordingCorpus(C, pool)
    splits=SplitCorpus.transcript_split(C, recordings)
    return C, splits
