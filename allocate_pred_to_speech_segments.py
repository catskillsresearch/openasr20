import numpy as np

def align_seg_words(seg_words):
    ([seg_start, seg_end], seg_wrds) = seg_words
    seg_duration=seg_end-seg_start
    n_seg_wrds=len(seg_wrds)
    word_duration=seg_duration//n_seg_wrds
    seg_duration, word_duration
    seg_word_boundaries=np.hstack([np.linspace(seg_start, seg_end-word_duration, n_seg_wrds).astype(int), [seg_end]])
    seg_aligned_wrds=[(seg_word_boundaries[i], seg_word_boundaries[i+1], seg_wrds[i]) for i in range(n_seg_wrds)]
    return seg_aligned_wrds

def align_segment_words(segment_words):
    return [z for y in [align_seg_words(x) for x in segment_words] for z in y]

def allocate_pred_to_speech_segments(prediction, speech_segments):
    pred_words=prediction.split(' ')
    n_words=len(pred_words)
    if n_words==0:
        return []
    segment_durations=np.diff(speech_segments)
    speech_duration=segment_durations.sum()
    segment_allocation=n_words*segment_durations/speech_duration
    words_per_segment=np.round(segment_allocation).T.astype(int)[0]
    # If count is under then add missing word to longest segment
    words_per_segment[np.where(words_per_segment==words_per_segment.max())[0][0]] += n_words-words_per_segment.sum()
    word_segment_boundaries=np.cumsum(np.hstack([[0],words_per_segment]))
    segment_words=list(zip(speech_segments.tolist(),
                           [pred_words[word_segment_boundaries[i]:word_segment_boundaries[i+1]]
                            for i in range(len(words_per_segment))]))
    return align_segment_words(segment_words)
