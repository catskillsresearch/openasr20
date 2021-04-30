import pickle

def pload(fn):
    print("loading", fn)
    with open(fn, 'rb') as f:
        return pickle.load(f)
