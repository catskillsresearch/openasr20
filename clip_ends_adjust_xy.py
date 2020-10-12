from clip_ends import clip_ends

def clip_ends_adjust_xy(audio, x, y):
    clip, (x1,y1) = clip_ends(audio)
    return clip, (x1+x, y1+x)

