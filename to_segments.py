import math
from padarray import padarray

def to_segments(V, window):
    recording_length = V.shape[0]
    n_segments=math.ceil(recording_length/window)
    segments=[]
    for i in range(n_segments):
        sample=V[i*window:(i+1)*window]
        if sample.shape[0] != window:
            sample=padarray(sample, window)
        segments.append(sample)
    return segments
