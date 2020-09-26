import numpy as np
from split_on_longest_median_low_energy_point import split_on_longest_median_low_energy_point
from AudioTranscriptionSample import AudioTranscriptionSample

def sample_split(artifact, target_max_sample_length):
    sound=artifact.source.value
    if sound.shape[0] <= target_max_sample_length:
        return [artifact]
    C=artifact.source.C
    text=artifact.target.value
    tokens=artifact.target.tokens
    n_text=len(text)
    n_tokens=len(tokens)
    left_audio,right_audio=split_on_longest_median_low_energy_point(C, sound, debug=None)
    left_size, right_size = (left_audio.shape[0], right_audio.shape[0])
    combined_size=left_size+right_size
    avg_samples_per_grapheme=combined_size/n_text
    left_pct=left_size/combined_size
    right_pct=right_size/combined_size
    text_pcts=np.array([len(tokens[i])/n_text for i in range(n_tokens)])
    text_pcts=text_pcts/text_pcts.sum()
    text_cum=np.cumsum(text_pcts)
    left_boundary=np.where(left_pct >= text_cum)[0][-1]
    left_tokens = tokens[0:left_boundary+1]
    left_text=' '.join(left_tokens)
    right_tokens = tokens[left_boundary+1:]
    right_text=' '.join(right_tokens)
    key = artifact.key
    left_sample = AudioTranscriptionSample(C, key, key+'L', None, left_audio, None, left_text)
    right_sample = AudioTranscriptionSample(C, key, key+'R', None, right_audio, None, right_text)
    return [left_sample, right_sample]
