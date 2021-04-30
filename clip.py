from normalize import normalize

def clip(y,level=0.1):
    y[y<level]=0
    return normalize(y)
