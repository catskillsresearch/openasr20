import tqdm
from SplitCorpus import SplitCorpus
from split_artifact import split_artifact

class SubSplitCorpus(SplitCorpus):
    
    def __init__(self, _corpus):
        C = _corpus.C
        df = _corpus.sample_statistics()
        median_words_in_sample=int(df[(df.Corpus=="Split Transcription") &
                                      (df.Units=="Length in words") &
                                      (df.Measurement=="Median")].Value.values[0])

        R = []
        problems=[]
        for artifact in tqdm.notebook.tqdm(_corpus.artifacts):
            try:
                R.append(split_artifact(C, median_words_in_sample, artifact))
            except:
                problems.append(artifact)
        split_artifacts = [item for sublist in R for item in sublist]
        super().__init__(C, split_artifacts)
        self.median_words_in_sample=median_words_in_sample
        self.problems = problems
