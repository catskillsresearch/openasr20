def chunk_size(fn):
    with open(fn, 'r') as f:
        return (len(f.read().strip().split(' ')), fn)
