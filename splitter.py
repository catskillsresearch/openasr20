from split_artifact import split_artifact

def splitter(task):
    (C, median_words_in_sample, artifact) = task
    try:
        return split_artifact(C, median_words_in_sample, artifact)
    except:
        return [artifact]
