import numpy as np

def normalize(A):
    A=np.copy(A)
    A=A-A.min()
    A=A/A.max()
    return A
