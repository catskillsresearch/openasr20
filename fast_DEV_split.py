from Cfg import Cfg
import sys
import tqdm
from glob import glob
from multiprocessing import Pool
from fast_DEV_split_file import fast_DEV_split_file

def fast_DEV_split(C):
    wavfns = glob(f'{C.build_dir}/audio/*.wav')
    tasks = [(C,fn) for fn in wavfns]
    with Pool(16) as pool:
        _ = list(tqdm.tqdm(pool.imap(fast_DEV_split_file, tasks), total=len(tasks)))

    # Check clip sizes
    splits=C.split_files()
    L=[x['t_seconds'] for x in splits]
    print("clip sizes: min {min(L)}, max{max(L)}, mean {sum(L)/len(L)}")
    if min(L) < 0.2:
        print("ERROR: some clips too small")
    if max(L) > 16.5:
        print("ERROR: some clips too large")

if __name__=="__main__":
    language = sys.argv[1]
    C = Cfg('NIST', 16000, language, 'dev')
    fast_DEV_split(C)
