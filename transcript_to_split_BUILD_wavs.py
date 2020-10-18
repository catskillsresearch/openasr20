# coding: utf-8

# Transcript to BUILD wavs

import os
from glob import glob
from tqdm import tqdm
from transcript_to_split_BUILD_wavs_file import transcript_to_split_BUILD_wavs_file

def transcript_to_split_BUILD_wavs(C):
    os.system(f'mkdir -p {C.build_dir}/audio_split')
    os.system(f'mkdir -p {C.build_dir}/silence_split')
    os.system(f'mkdir -p {C.build_dir}/transcription_split')
    transcripts = list(sorted(glob(f'{C.stage}/openasr20_{C.language}/build/transcription/*.txt')))
    for transcript_file in tqdm(transcripts):
        transcript_to_split_BUILD_wavs_file(C, transcript_file)

if __name__=="__main__":
    from Cfg import Cfg
    from pprint import pprint
    C = Cfg('NIST', 16000, 'pashto', 'build', '001')
    transcript_to_split_BUILD_wavs(C)
