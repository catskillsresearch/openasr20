import os, sys
from Cfg import Cfg
from tqdm.auto import tqdm
from glob import glob

phase=sys.argv[1]
runid=sys.argv[2]

scoring=f'scoring/{phase}/{runid}'
os.system(f'mkdir -p {scoring}')
C = Cfg('NIST', 16000, 'amharic', phase)
splits=C.split_files()
recordings=list(set(['_'.join(split['key'][0:7]) for split in splits]))
for recording in tqdm(recordings):
    ref_file=f'NIST/openasr20_{C.language}/{C.phase}/transcription_stm/{recording}.stm'
    ctm_file=f'ship/{C.language}/{runid}/{recording}.ctm'
    cmd=f'./SCTK/bin/sclite -r {ref_file} stm -h {ctm_file} ctm -F -D -O {scoring} -o sum rsum pralign prf -e utf-8'
    os.system(cmd)
