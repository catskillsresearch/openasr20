from recursive_split import recursive_split
from midpoint import midpoint

def optimal_split(t_lower, t_upper, n_words, audio, window = 500, min_gap = 10):

    S_lower = recursive_split(audio, window, t_lower, min_gap); 
    n_lower=len(S_lower)
    if n_lower==n_words:
        return S_lower

    S_upper = recursive_split(audio, window, t_upper, min_gap)
    n_upper=len(S_upper)
    if n_upper==n_words:
        return S_upper

    for i in range(30):

        if n_lower==n_upper:
            return S_lower

        t_mid=midpoint(t_lower, t_upper)
        S_mid = recursive_split(audio, window, t_mid, min_gap)
        n_mid=len(S_mid)

        if n_mid == n_words:
            return S_mid

        if n_mid > n_words:
            (S_upper, n_upper, t_upper) = (S_mid, n_mid, t_mid)
        else:
            (S_lower, n_lower, t_lower) = (S_mid, n_mid, t_mid)

    return S_lower
