# coding: utf-8

# Transcript to BUILD wavs

from glob import glob
from tqdm import tqdm
from transcript_to_split_BUILD_wavs_file import transcript_to_split_BUILD_wavs_file

def transcript_to_split_BUILD_wavs(C):
    transcripts = list(sorted(glob(f'{C.stage}/openasr20_{C.language}/build/transcription/*.txt')))
    audio_files=[x.replace('/transcription/', '/audio/').replace('.txt','.wav') for x in transcripts]
    for transcript_file in tqdm(transcripts):
        transcript_to_split_BUILD_wavs_file(C, transcript_file)
