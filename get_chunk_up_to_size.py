def get_chunk_up_to_size(L,goal_length_in_seconds,sample_rate):
    size=0
    for i, (start,end) in enumerate(L):
        size += (end-start)/sample_rate
        if size >= goal_length_in_seconds:
            i=max(0,i-1)
            break
    return L[0:i+1], L[i+1:]
