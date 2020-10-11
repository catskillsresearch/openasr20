import numpy as np

def eos_trim(v):
    try:
        return ''.join(v[0:np.where(v=='<eos>')[0][0]])
    except:
        return ''.join(v)
