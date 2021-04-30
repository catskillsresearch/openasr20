import os, sys
from Cfg import Cfg
from tqdm.auto import tqdm
from glob import glob

language=sys.argv[1]
phase=sys.argv[2]
release=sys.argv[3]
C = Cfg('NIST', 16000, language, phase, release)

scoring=f'scoring/{language}/{phase}/{release}'
os.system(f'mkdir -p {scoring}')
splits=C.split_files()
recordings=list(set(['_'.join(split['key'][0:7]) for split in splits]))
for recording in tqdm(recordings):
    ref_file=f'NIST/openasr20_{C.language}/{C.phase}/transcription_stm/{recording}.stm'
    ctm_file=f'ship/{C.language}/{phase}/{release}/{recording}.ctm'
    cmd=f'./SCTK/bin/sclite -r {ref_file} stm -h {ctm_file} ctm -F -D -O {scoring} -o sum rsum pralign prf -e utf-8'
    print(cmd)
    os.system(cmd)
