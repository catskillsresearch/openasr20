import os
from glob import glob
from tqdm import tqdm
from advanced_sample_treatment_for_BUILD_file import advanced_sample_treatment_for_BUILD_file

def advanced_sample_treatment_for_BUILD(C):
#    if os.path.exists(C.nr_dir):  return
    os.system(f"mkdir -p {C.nr_dir}")
    gold_fns=list(sorted(glob(f'{C.split_dir}/*.wav')))
    for fn in tqdm(gold_fns):
        advanced_sample_treatment_for_BUILD_file(C, fn)

if __name__=="__main__":
    from config import C
    advanced_sample_treatment_for_BUILD(C)
