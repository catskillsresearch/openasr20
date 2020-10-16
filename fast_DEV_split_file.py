from Cfg import Cfg
import soundfile as sf
import pyAudioAnalysis.audioSegmentation as aS
from normalize import normalize

def fast_DEV_split_audio(C, audio, root, weight, fn):
    # st_win, st_step:    window size and step in seconds
    # smoothWindow:     (optinal) smooth window (in seconds)
    # weight:           (optinal) weight factor (0 < weight < 1) the higher, the more strict
    (st_win, st_step) = 0.05, 0.05
    smoothWindow = 1
    segments = aS.silence_removal(audio, C.sample_rate, st_win, st_step, smoothWindow, weight)
    min_length=0.185
    regions=[(int(C.sample_rate*x), int(C.sample_rate*y)) for x,y in segments] # if y-x >= min_length]
    too_short = 0.2
    for start, end in regions:
        goal_length_in_seconds = 16.5
        seconds = (end-start)/C.sample_rate
        if seconds < too_short:
            continue
        if seconds > goal_length_in_seconds:
            if weight < 0.9:
                fast_DEV_split_audio(C, audio[start:end], root, 0.9, fn)
            else: # Lose information
                end_clipped = start + int(sample_rate*16.5)
                clip_fn = f"{fn[:-4].replace('/audio/', '/audio_split/')}_{C.sample_rate}_{start+root}_{end_clipped+root}.wav"
                sf.write(clip_fn, audio[start:end_clipped], C.sample_rate)
                print("FAIL", clip_fn, start, end, end_clipped)
        else:
            clip_fn = f"{fn[:-4].replace('/audio/', '/audio_split/')}_{C.sample_rate}_{start+root}_{end+root}.wav"
            sf.write(clip_fn, audio[start:end], C.sample_rate)

def fast_DEV_split_file(task, stop_early=False):
    C, fn = task
    (audio,Fs) = sf.read(fn)
    audio = normalize(audio)
    return fast_DEV_split_audio(C, audio, 0, 0.5, fn)

if __name__=="__main__":
    C = Cfg('NIST', 16000, 'amharic', 'dev')
    fn = 'NIST/openasr20_amharic/dev/audio/BABEL_OP3_307_69153_20140624_193324_inLine.wav'
    from pprint import pprint
    task=(C,fn)
    pprint(fast_DEV_split_file(task))
