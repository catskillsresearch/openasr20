import numpy as np

@np.vectorize
def to_TRG(TRG, x):
    return TRG.vocab.itos[x]
