import sys
from Cfg import Cfg

def check_clip_sizes(C):
    # Check clip sizes
    splits=C.split_files()
    L=[x['t_seconds'] for x in splits]
    print(f"clip sizes: min {min(L)}, max {max(L)}, mean {sum(L)/len(L)}")
    if min(L) < 0.2:
        print("ERROR: some clips too small")
    if max(L) > 16.5:
        print("ERROR: some clips too large")

if __name__=="__main__":
    language = sys.argv[1]
    C = Cfg('NIST', 16000, language, 'dev')
    check_clip_sizes(C)
