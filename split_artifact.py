import numpy as np
from recursive_split import recursive_split
from AudioTranscriptionSample import AudioTranscriptionSample

def split_artifact(config, median_words_in_sample, artifact):
    if artifact.target.n_words <= median_words_in_sample:
        return [artifact]
    audio=artifact.source.value
    A=recursive_split(audio, clip=0.0001)
    n_chunks=len(A)
    tokens=artifact.target.tokens
    n_tokens=len(tokens)
    token_graphemes=np.array([len(x) for x in tokens])
    n_token_graphemes=token_graphemes.sum()
    token_weights=token_graphemes/n_token_graphemes
    token_weights=token_weights/token_weights.sum()
    groups=(n_chunks-np.round(np.hstack([[0], np.cumsum(token_weights[::-1])])*n_chunks).astype(int))[::-1]
    B=[np.hstack(A[groups[i]:groups[i+1]]) for i in range(n_tokens)]
    phrase_groups=np.arange(0,n_tokens,median_words_in_sample)
    if phrase_groups[-1] != n_tokens:
        phrase_groups=np.append(phrase_groups, [n_tokens])
    phrase_groups=(phrase_groups+np.linspace(0,n_tokens % median_words_in_sample, len(phrase_groups)).astype(int))
    C=[np.hstack(B[phrase_groups[i]:phrase_groups[i+1]]) for i in range(phrase_groups.shape[0]-1)]
    T=[' '.join(tokens[phrase_groups[i]:phrase_groups[i+1]]) for i in range(phrase_groups.shape[0]-1)]
    key = artifact.key
    S = [AudioTranscriptionSample(config, key, f'{key}_{i}', None, audio, None, text)
         for i, (audio, text) in enumerate(zip(C, T))]
    return S