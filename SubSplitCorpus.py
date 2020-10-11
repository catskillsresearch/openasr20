import tqdm
from SplitCorpus import SplitCorpus
from splitter import splitter
from clipper import clipper

class SubSplitCorpus(SplitCorpus):
    
    def __init__(self, pool, _corpus, min_words=2):
        C = _corpus.C
        df = _corpus.sample_statistics()
        median_words_in_sample=int(df[(df.Corpus=="Split Transcription") &
                                      (df.Units=="Length in words") &
                                      (df.Measurement=="Median")].Value.values[0])
        R = []
        self.problems=[]
        goal = min(min_words, median_words_in_sample)
        tasks = [(artifact,goal) for artifact in _corpus.artifacts]
        splits = list(tqdm.tqdm(pool.imap(splitter, tasks), total=len(tasks)))
        for split in splits:
            if split[0]:
                R.append(split[1])
            else:
                self.problems.append(split[1])
        artifacts = [item for sublist in R for item in sublist]
        good = []
        for artifact in artifacts:
            if artifact.good_split():
                good.append(artifact)
            else:
                self.problems.append(artifact)
        good = list(tqdm.tqdm(pool.imap(clipper, good), total=len(good)))
        bad = [x for x in good if not x.good_split()]
        good = [x for x in good if x.good_split()]
        self.problems.extend(bad)
        super().__init__(C, good)
        self.median_words_in_sample=median_words_in_sample
