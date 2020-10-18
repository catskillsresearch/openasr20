from epoch_time import epoch_time
from matplotlib.pylab import *
from padarray import padarray
from pathlib import Path
from to_samples import to_samples
from torch.utils.data import TensorDataset, DataLoader
from txt_to_stm import txt_to_stm
import audioread, librosa, os
import numpy as np
import pandas as pd
import soundfile as sf

def transcript_to_split_BUILD_wavs_file(C, transcript_file):
    language=C.language
    no_Q = (language == 'cantonese')
    audio_file = transcript_file.replace('/transcription/', '/audio/').replace('.txt','.wav')
    if not os.path.exists(audio_file):
        print('missing', audio_file)
        return
        
    # Load audio
    file = "_".join(os.path.basename(transcript_file).split("_")[:-1])
    channel = os.path.basename(transcript_file).split("_")[-1].split(".")[-2]
    transcript_df = pd.read_csv(transcript_file, sep = "\n", header = None, names = ["content"])
    result = txt_to_stm(transcript_df, file, channel, no_Q)
    x_np,sr=librosa.load(audio_file, sr=C.sample_rate)
    
    # Split silence
    silence=[(float(x[-2]), float(x[-1])) for x in result if len(x)==5]
    silence_segments=[(int(a*C.sample_rate), int(b*C.sample_rate)) for (a,b) in silence]
    for i, (lower, upper) in enumerate(silence_segments):
        audio_split_file=f"{audio_file[0:-4].replace('/audio/','/silence_split/')}_{lower}_{upper}.wav"
        sf.write(audio_split_file, x_np[lower:upper], C.sample_rate)

    # Split audio
    speech=[(float(x[-3]), float(x[-2]), x[-1]) for x in result if len(x)==6]
    speech_segments=[(int(a*C.sample_rate), int(b*C.sample_rate), words)
                     for (a,b,words) in speech
                     if 'IGNORE' not in words]
    for i, (lower, upper, words) in enumerate(speech_segments):
        audio_split_file=f"{audio_file[0:-4].replace('/audio/','/audio_split/')}_{lower}_{upper}.wav"
        transcript_split_file=f"{transcript_file[0:-4].replace('/transcription/','/transcription_split/')}_{lower}_{upper}.txt"
        sf.write(audio_split_file, x_np[lower:upper], C.sample_rate)
        with open(transcript_split_file,'w') as f:
            f.write(words)
