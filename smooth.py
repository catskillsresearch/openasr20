import numpy as np
from normalize import normalize

def smooth(y, w):
    box = np.ones(w)/w
    y_smooth = np.convolve(y, box, mode='same')
    return normalize(y_smooth)
