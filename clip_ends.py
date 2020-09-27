import numpy as np
from normalize import normalize

def clip_ends(audio, clip=.0005):
    audio2 = audio**2
    x = 0
    y = audio2.shape[0]

    try:
        start_x=np.where(audio2 > clip)[0][0]
        x += start_x
    except:
        pass

    audio3=audio2[::-1]
    try:
        start_y=np.where(audio3 > clip)[0][0]
        y -= start_y
    except:
        pass

    x -= 100
    if x < 0:
        x = 0

    y += 100

    return audio[x:y]
