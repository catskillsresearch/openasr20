import numpy as np

def to_samples(V, window, N=10000):
    i = 0
    samples=[]
    chunk_interval=int(V.shape[0]/N)
    for i in range(N):
        sample=V[i*chunk_interval:i*chunk_interval+window]
        if sample.shape[0] != window:
            break
        samples.append(sample)
    return samples
