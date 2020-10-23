import numpy as np
import librosa
import matplotlib.pylab as plt
from transcribe import transcribe
from collect_false import collect_false

def transcription(C, model, audio, max_duration):
    cuts=[0]
    preds=[]
    while True:
        size=audio.shape[0]
        if size == 0:
            break
        max_samples=int(max_duration*C.sample_rate)
        min_samples=int(0.2*C.sample_rate)
        if size > max_samples:
            for cutoff in np.linspace(-80,-8,200):
                T=audio.shape[0]/C.sample_rate
                S = librosa.feature.melspectrogram(y=audio, sr=C.sample_rate, n_mels=64, fmax=8000)
                S_dB = librosa.power_to_db(S, ref=np.max)
                s_dB_mean=np.mean(S_dB,axis=0)
                speech_q=(s_dB_mean>cutoff)
                silences=T*collect_false(speech_q)/len(speech_q)
                if 1:
                    print(f'cutoff {cutoff} #silences {len(silences)}: {silences[0:4]}')
                S2=[(x,y) for x,y in silences if max_duration >= y > 0.05]
                S2a=[(x,y) for x,y in silences if 6 >= y > 0.05]
                if len(S2a):
                    S2=S2a
                if len(S2):
                    break
                cutoff += 1
                if cutoff > -18:
                    if 0:
                        plt.figure(figsize=(60,4))
                        plt.plot(s_dB_mean)
                        plt.show()
                        plt.figure(figsize=(60,4))
                        plt.plot(audio);
                    raise ValueError('couldnt split clip')
            S3=int(S2[-1][-1]*C.sample_rate)
        else:
            S3=size
        clip=audio[0:S3]
        pred=transcribe(C, model, clip)
        cuts.append(S3)
        preds.append(pred)
        if False and pred != '':
            print(f"sample size in seconds {S3/C.sample_rate} pred {pred} :: {unidecode(pred)}")
            play(clip)
        audio=audio[S3:]
        if audio.shape[0] < min_samples:
            break

    times=np.cumsum(cuts)/C.sample_rate
    transcript=[(times[i], times[i+1], preds[i]) for i in range(len(preds)) if preds[i]]
    
    return transcript
