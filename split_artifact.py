import numpy as np
from AudioTextSample import AudioTextSample
from aggressive_clip_ends import aggressive_clip_ends
from optimal_split import optimal_split

def split_artifact(artifact, median_words_in_sample, window=500):
    config = artifact.target.C
    if artifact.target.n_words <= median_words_in_sample:
        return [artifact]
    if artifact.source.n_samples < 3*config.sample_rate:  # Under 3 seconds is fine
        return [artifact]
    A=optimal_split(0.0001, 0.6, artifact.target.n_words, artifact.source.value, window=window)
    n_chunks=len(A)
    tokens=artifact.target.tokens
    n_tokens=len(tokens)
    token_graphemes=np.array([len(x) for x in tokens])
    n_token_graphemes=token_graphemes.sum()
    token_weights=token_graphemes/n_token_graphemes
    token_weights=token_weights/token_weights.sum()
    groups=(n_chunks-np.round(np.hstack([[0], np.cumsum(token_weights[::-1])])*n_chunks).astype(int))[::-1]
    groups=list(dict.fromkeys(groups))
    if len(groups) < 2:
        return [artifact]
    B=[np.hstack(A[groups[i]:groups[i+1]]) for i in range(len(groups)-1)]
    if len(B)==1:
        C = B
        T = ' '.join(tokens)
    else:
        phrase_groups=np.arange(0,n_tokens,median_words_in_sample)
        if phrase_groups[-1] != n_tokens:
            phrase_groups=np.append(phrase_groups, [n_tokens])
        phrase_groups=(phrase_groups+np.linspace(0,n_tokens % median_words_in_sample, len(phrase_groups)).astype(int))
        C=[np.hstack(B[phrase_groups[i]:phrase_groups[i+1]]) for i in range(phrase_groups.shape[0]-1)]
        T=[' '.join(tokens[phrase_groups[i]:phrase_groups[i+1]]) for i in range(phrase_groups.shape[0]-1)]
        key = artifact.key
    S = [AudioTextSample(config, artifact, i, aggressive_clip_ends(audio, config.sample_rate)[0], text)
         for i, (audio, text) in enumerate(zip(C, T))]
    return S
