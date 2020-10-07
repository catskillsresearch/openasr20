import tqdm
from SplitCorpus import SplitCorpus
from split_artifact import split_artifact

class SubSplitCorpus(SplitCorpus):
    
    def __init__(self, _corpus, min_words=2):
        C = _corpus.C
        df = _corpus.sample_statistics()
        median_words_in_sample=int(df[(df.Corpus=="Split Transcription") &
                                      (df.Units=="Length in words") &
                                      (df.Measurement=="Median")].Value.values[0])

        R = []
        problems=[]
        for artifact in tqdm.notebook.tqdm(_corpus.artifacts):
            try:
                R.append(split_artifact(artifact, min(min_words, median_words_in_sample)))
            except:
                problems.append(artifact)
        split_artifacts = [item for sublist in R for item in sublist]
        good=[x for x in split_artifacts if x.target.n_graphemes > 0 and x.source.n_seconds <= 4]
        bad=[x for x in split_artifacts if x.target.n_graphemes ==0 or x.source.n_seconds > 4]
        for x in tqdm.notebook.tqdm(good):
            x.aggressively_clip()
        super().__init__(C, good)
        self.median_words_in_sample=median_words_in_sample
        self.problems = bad
