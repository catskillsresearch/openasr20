from sample_split import sample_split

def split_to_target_sample_length(artifact, target_max_sample_length, split_on = 'tokens'):
    S=sample_split(artifact, target_max_sample_length, split_on)
    if len(S)==1:
        return S
    else:
        SL=split_to_target_sample_length(S[0], target_max_sample_length)
        SR=split_to_target_sample_length(S[1], target_max_sample_length)
        return SL+SR
