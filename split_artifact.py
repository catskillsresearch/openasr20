import numpy as np
from AudioTextSample import AudioTextSample
from aggressive_clip_ends import aggressive_clip_ends
from optimal_split import optimal_split
from weights import weights

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
    if n_chunks == n_tokens:
        T = tokens
    else:
        if n_chunks==1:
            T = [' '.join(tokens)]
        else:
            token_weights=weights(tokens)*n_chunks
            cum = 0
            token_to_chunk={i:[] for i in range(n_chunks)}
            token_to_chunk
            next_chunk=1
            for token,w in zip(tokens,token_weights):
                cum += w
                if cum < next_chunk:
                    token_to_chunk[next_chunk-1].append(token)
                else:
                    token_to_chunk[next_chunk].append(token)
                    next_chunk = min(next_chunk+1, n_chunks-1)

            T=[' '.join(token_to_chunk[i]) for i in range(n_chunks)]
    S = [AudioTextSample(config, artifact, i, aggressive_clip_ends(audio, config.sample_rate)[0], text)
         for i, (audio, text) in enumerate(zip(A, T))]
    return S
