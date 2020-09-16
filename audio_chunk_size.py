import os

def audio_chunk_size(fn):
    start,end = os.path.basename(fn)[0:-4].split('_')[-2:]
    return (float(end)-float(start), fn)
