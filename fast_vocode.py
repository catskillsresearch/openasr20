import sys, os, tqdm
from glob import glob
from multiprocessing import Pool
from fast_vocode_file import fast_vocode_file

def fast_vocode():
    os.system('mkdir -p NIST/openasr20_amharic/build/audio_vocode')
    tasks = glob(f'NIST/openasr20_amharic/build/audio_split/*.wav')
    for task in tqdm.tqdm(tasks):
        fast_vocode_file(task)

if __name__=="__main__":
    fast_vocode()
