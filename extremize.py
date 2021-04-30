def extremize(y,level=0.1):
    y[y<level]=0
    y[y>=level]=1
    return y
