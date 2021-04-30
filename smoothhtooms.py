from smooth import smooth

def smoothhtooms(A,w):
    A1=smooth(A,w)
    A2=smooth(A1[::-1],w)
    return A2[::-1]
