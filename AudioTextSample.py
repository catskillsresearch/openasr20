from Sample import Sample
from AudioArtifact import AudioArtifact
from PhraseArtifact import PhraseArtifact
import numpy as np
from aggressive_clip_ends import aggressive_clip_ends
from optimal_split import optimal_split
from weights import weights

class AudioTextSample(Sample):

    def __init__(self, _config,  _key, _audio, _transcription):
        super().__init__(_key,
                         AudioArtifact(_config, _audio),
                         PhraseArtifact(_config, _transcription))

    def aggressively_clip(self):
        self.source.aggressively_clip()

    def display(self):
        print('KEY', self.key)
        print('SOURCE')
        self.source.display(self.target.romanized)
        print('TARGET')
        self.target.display()
        print()

    def good_split(self):
        if self.target.n_graphemes > 0:
            if self.source.n_seconds <= 4:
                if self.source.n_samples > 370:
                    return True
        return False

    def split(self, median_words_in_sample, window=500):
        config = self.target.C
        if self.target.n_words <= median_words_in_sample:
            return [AudioTextSample(config, self.key+(0,), self.source.value, self.target.value)]
        if self.source.n_samples < 3*config.sample_rate:  # Under 3 seconds is fine
            return [AudioTextSample(config, self.key+(0,), self.source.value, self.target.value)]
        A=optimal_split(0.0001, 0.6, self.target.n_words, self.source.value, window=window)
        n_chunks=len(A)
        tokens=self.target.tokens
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
        S = [AudioTextSample(config, self.key+(i,), aggressive_clip_ends(audio, config.sample_rate)[0], text)
             for i, (audio, text) in enumerate(zip(A, T))]
        return S
