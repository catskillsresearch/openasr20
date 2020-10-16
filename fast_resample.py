import sys
import tqdm
from glob import glob
from multiprocessing import Pool
from fast_resample_file import fast_resample_file

def fast_resample():
    tasks = glob(f'NIST/*/build/audio/*.wav')
    with Pool(16) as pool:
        _ = list(tqdm.tqdm(pool.imap(fast_resample_file, tasks), total=len(tasks)))

if __name__=="__main__":
    fast_resample()
