import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
from transcribe import transcribe
from collect_false import collect_false
from predicted_segment_transcript import predicted_segment_transcript
from plot_predicted_segments import plot_predicted_segments
from matplotlib.pyplot import *
from smoothhtooms import smoothhtooms

def listen_and_transcribe(C, model, max_duration, gold, audio, debug=False):
    audio /= max(abs(audio.min()), abs(audio.max()))
    size=audio.shape[0]
    T=size/C.sample_rate
    X=np.arange(size)/C.sample_rate
    Z=np.zeros(size)
    S = librosa.feature.melspectrogram(y=audio, sr=C.sample_rate, n_mels=64, fmax=8000)
    dt_S=T/S.shape[1]
    samples_per_spect=int(dt_S*C.sample_rate)
    S_dB = librosa.power_to_db(S, ref=np.max)
    s_dB_mean=np.mean(S_dB,axis=0)
    max_samples=int(max_duration/dt_S)
    min_samples=1
    pred=[]
    cutoffs = np.linspace(-80,-18,200)
    max_read_head=s_dB_mean.shape[0]
    max_read_head, max_samples, min_samples
    read_head=0
    transcriptions=[]
    read_heads=[]
    read_heads=[read_head]
    while read_head < max_read_head:
        if debug: print(read_head, read_heads)
        finished = False
        while not finished and read_head < max_read_head:
            if debug: print(f"READ_HEAD {read_head}")
            for cutoff in cutoffs:
                speech_q=(s_dB_mean[read_head:]>cutoff)
                silences=collect_false(speech_q)
                silences=[(x,y) for x,y in silences if x != y and y-x > min_samples]
                n_silences = len(silences)
                if debug: print(f'cutoff {cutoff} #silences {n_silences}: {silences[0:10]}')
                if n_silences==0:
                    continue
                elif silences[0][0] == 0 and silences[0][1] != 0:
                    read_head +=silences[0][1]
                    if debug: print("advance past silence", read_head, silences[0][1])
                    break
                elif silences[0][0] > max_samples:
                    continue
                else:
                    silences=[(x,y) for x,y in silences if x <= max_samples]
                    if not len(silences):
                        continue
                    start_at = read_head
                    stop_at= read_head + silences[0][0]
                    read_head = stop_at
                    if debug: print("stop to read", start_at, stop_at)
                    finished = True
                    break
            if not finished:
                display_start=read_head*samples_per_spect
                display_end=display_start+max_samples
                if debug: 
                    plot(X[display_start:display_end],audio[display_start:display_end])
                    title(f'NO REMAINING SILENCE')
                    show()
                start_at = read_head
                stop_at = min(max_read_head, read_head + max_samples)
                read_head = stop_at
                finished = True

        read_heads.append(read_head)
        start=start_at*samples_per_spect
        end=start_at+stop_at*samples_per_spect
        display_start=max(0, start-5*C.sample_rate)
        display_end=end+5*C.sample_rate
        smooth_abs=smoothhtooms(np.abs(audio[start:end]), 100)
        smooth_abs_max=smooth_abs.max()
        if smooth_abs_max < 0.05:
            if debug: 
                figure(figsize=(20,6))
                plot(X[display_start:display_end],audio[display_start:display_end])
                plot(X[start:end],Z[start:end],color='red',linewidth=3)
                title(f'SKIPPED max {smooth_abs_max}')
                show()
        else:
            if debug: 
                figure(figsize=(20,6))
                plot(X[display_start:display_end],audio[display_start:display_end])
                plot(X[start:end],Z[start:end],color='red',linewidth=3)
                title('INCLUDED')
                show()
            try:
                segment_transcript, timeline, normalized_power, speech_mask, clip_audio=\
                    predicted_segment_transcript(C, model, audio, start, end, s_dB_mean, samples_per_spect, dt_S)
                transcriptions.extend(segment_transcript)
                if debug: 
                    plot_predicted_segments(timeline, normalized_power, speech_mask, segment_transcript, gold)
                    show()
            except:
                if debug: 
                    print("empty translation")
    transcriptions = [(time, time+duration, pred) for time, duration, pred in transcriptions]
    return transcriptions

if __name__=="__main__":
    from multiprocessing import Pool
    from Cfg import Cfg
    from load_pretrained_model import load_pretrained_model
    from RecordingCorpus import RecordingCorpus
    C = Cfg('NIST', 16000, 'vietnamese', 'build') 
    with Pool(16) as pool:
        recordings = RecordingCorpus(C, pool)
    artifacts=recordings.artifacts
    artifact=[x for x in artifacts if x.key[1]=='BABEL_BP_107_54621_20120421_132410_outLine'][0]
    audio=artifact.source.value
    gold=[x[3:] for x in artifact.target.value if len(x)==6]
    gold=[(float(start), words) for start, finish, words in gold]
    model = load_pretrained_model(C, 0)
    max_duration=20
    transcriptions=listen_and_transcribe(C, model, max_duration, gold, audio)
