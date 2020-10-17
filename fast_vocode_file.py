import sys
sys.path.append('/home/catskills/Desktop/openasr20/Python-WORLD')

import librosa
from world import main
from normalize import normalize
import soundfile as sf

def fast_vocode_file(wav_path):
    sample_rate=16000
    x,fs=librosa.load(wav_path,sr=sample_rate)
    vocoder = main.World()
    dat = vocoder.encode(fs, x, f0_method='harvest', is_requiem=True)
    dat = vocoder.decode(dat)
    result=normalize(dat['out'])
    new_path=wav_path.replace('/audio/', '/audio_vocode/')
    sf.write(new_path, result, sample_rate)
