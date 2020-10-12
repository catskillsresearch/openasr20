from get_chunk_up_to_size import get_chunk_up_to_size

def chunks_of_size(L,goal_length_in_seconds,sample_rate):
    G=[]
    R=L
    for i in range(len(L)):
        chunk, R = get_chunk_up_to_size(R,goal_length_in_seconds,sample_rate)
        G.append((chunk[0][0], chunk[-1][-1]))
        if not R:
            break
    return G
