import numpy as np

def trim_ends(x0,y0,audio,Fs):
    x = int(x0*Fs)
    y = int(y0*Fs)
    audio2=audio[x:y]
    clip=.0005
    try:
        start_x=np.where(audio2 > clip)[0][0]
        x += start_x
    except:
        start_x = 0
    audio3=audio2[::-1]
    try:
        start_y=np.where(audio3 > clip)[0][0]
        y -= start_y
    except:
        start_y = 0
    if start_x or start_y:
        x = x/Fs
        y = y/Fs
    else:
        x = x0
        y = y0
    return (x,y)
