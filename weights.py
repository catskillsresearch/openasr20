import numpy as np

def weights(tokens):
    W=np.array([len(token) for token in tokens])
    return W/W.sum()
