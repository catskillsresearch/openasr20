import numpy as np
import librosa
from collect_false import collect_false

def power_split(C, audio, max_duration=33):
    T=np.arange(audio.shape[0])/C.sample_rate
    S = librosa.feature.melspectrogram(y=audio, sr=C.sample_rate, n_mels=64, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    s_dB_mean=np.mean(S_dB,axis=0)
    db_grid = np.linspace(-60,-75,40)
    for cutoff in db_grid:
        speech_q=(s_dB_mean>cutoff)
        TSQ=T[-1]*np.arange(len(speech_q))/len(speech_q)
        silences=T[-1]*collect_false(speech_q)/len(speech_q)
        for gap in np.linspace(2,0.01,20):
            pauses=[(x,y) for x,y in silences if y-x > gap and int(y*C.sample_rate) < audio.shape[0]]
            cuts=np.array([int(C.sample_rate*(x+y)/2) for x,y in pauses if x != 0.0])
            sizes=np.diff(np.hstack([[0],cuts]))/C.sample_rate
            if sizes.max() < max_duration:
                return cuts, gap, T
    raise ValueError("clip too large")
