from split_on_longest_median_low_energy_point import split_on_longest_median_low_energy_point

def recursive_split(audio, window = 100, threshold = 0.3, min_gap = 10, clip=0.0005):
    left,right=split_on_longest_median_low_energy_point(audio, window, threshold, min_gap, clip)
    if right is None:
        return [left]
    else:
        return recursive_split(left, window, threshold, min_gap)+recursive_split(right, window, threshold, min_gap, clip)
