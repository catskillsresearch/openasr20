import numpy as np

# https://stackoverflow.com/questions/22592764/numpy-detection-of-region-borders

def contiguous_regions(condition):
    """Finds contiguous True regions of the 1D boolean array "condition".
    Returns a 2D array where the first column is the start index of the region
    and the second column is the end index."""
    # Find the indicies of changes in "condition"
    idx = np.flatnonzero(np.diff(condition)) + 1

    # Prepend or append the start or end indicies to "idx"
    # if there's a block of "True"'s at the start or end...
    if condition[0]:
        idx = np.append(0, idx)
    if condition[-1]:
        idx = np.append(idx, len(condition))

    return idx.reshape(-1, 2)
