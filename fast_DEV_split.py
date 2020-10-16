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

if __name__=="__main__":
    language = sys.argv[1]
    C = Cfg('NIST', 16000, language, 'dev')
    fast_DEV_split(C)
