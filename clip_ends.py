import numpy as np
from copy import deepcopy

def clip_ends(_audio, clip=.005):
    audio = deepcopy(_audio)
    
    audio2 = np.abs(audio)
    x = 0
    y = audio2.shape[0]

    try:
        start_x=np.where(audio2 > clip)[0][0]
        x += start_x
    except:
        audio[0:]=0
        return (audio, audio[0:0])

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

    audio[0:x] = 0
    audio[y:]=0
    
    return (audio, audio[x:y])
