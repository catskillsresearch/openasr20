from split_on_longest_median_low_energy_point import split_on_longest_median_low_energy_point

def recursive_split(audio, window = 100, threshold = 0.3, min_gap = 10, clip=0.0005):
    [Left, Right]=split_on_longest_median_low_energy_point(audio, window, threshold, min_gap, clip)
    (left,_,_) = Left
    if Right is None:
        return [left]
    else:
        (right,_,_) = Right
        return recursive_split(left, window, threshold, min_gap)+recursive_split(right, window, threshold, min_gap, clip)
