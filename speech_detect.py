import json, os
import soundfile as sf
import pyAudioAnalysis.audioSegmentation as aS
from trim_ends import trim_ends

def speech_detect(task):
    (wavfn,resfn) = task
    if os.path.isfile(resfn):
        return
    (audio,Fs) = sf.read(wavfn)
    # st_win, st_step:    window size and step in seconds
    # smoothWindow:     (optinal) smooth window (in seconds)
    # weight:           (optinal) weight factor (0 < weight < 1) the higher, the more strict
    (st_win, st_step) = 0.05, 0.05
    smoothWindow = 1
    weight = 0.05
    try:
        segments = aS.silenceRemoval(audio, Fs, st_win, st_step, smoothWindow, weight)
    except:
        print("ERROR", wavfn)
        with open(resfn, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=4)
        return
    segments = [(x,y) for x,y in segments if y-x >= 1/20.]       
    segments = [trim_ends(x,y,audio,Fs) for x,y in segments]
    with open(resfn, 'w', encoding='utf-8') as f:
        json.dump(segments, f, ensure_ascii=False, indent=4)
