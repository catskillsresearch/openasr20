import numpy as np
from itertools import groupby
from operator import itemgetter

def mask_boundaries(mask):
    groups = [[i for i, _ in group]
              for key, group in groupby(enumerate(mask), key=itemgetter(1))
              if key]
    boundaries=[(x[0],x[-1]) for x in groups]
    return np.array(boundaries)
