# coding: utf-8

import numpy as np
from itertools import groupby
from operator import itemgetter

def collect_false(silence_mask):
    groups = [[i for i, _ in group] for key, group in groupby(enumerate(silence_mask), key=itemgetter(1)) if not key]
    boundaries=[(x[0],x[-1]) for x in groups]
    return np.array(boundaries)
