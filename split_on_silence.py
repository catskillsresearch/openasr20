from good_large_split_unstripped import good_large_split_unstripped
from chunks_of_size import chunks_of_size
from clip_ends_adjust_xy import clip_ends_adjust_xy
from normalize import normalize

def split_on_silence(audio, t_lower, t_upper, window, min_gap, sample_rate, goal_length_in_seconds):
    L=good_large_split_unstripped(audio, t_lower, t_upper, window, min_gap, sample_rate, goal_length_in_seconds)
    G=chunks_of_size(L,goal_length_in_seconds,sample_rate)
    A=[clip_ends_adjust_xy(audio[x:y],x,y) for x,y in G]
    A1=[x for x in A if x[0].shape[0] > 0]
    A2=[(normalize(audio),(x,y)) for audio,(x,y) in A1]
    return A2
