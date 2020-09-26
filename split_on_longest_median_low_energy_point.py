import numpy as np
from aggressive_clip_ends import aggressive_clip_ends
from unidecode import unidecode
from itertools import groupby
from operator import itemgetter
import noisereduce as nr
from normalize import normalize
from IPython.display import Audio
from show_sound import show_sound

def split_on_longest_median_low_energy_point(C, sound, window = 100, threshold = 0.3, min_gap = 10, debug=None):
    audio_moving=np.convolve(np.abs(sound), np.ones((window,))/window, mode='same') 
    if debug:
        print(f'average energy over {window} sample window')
        plt.figure(figsize=(40,4))
        plt.plot(sound)
        plt.plot(audio_moving);
        plt.title(unidecode(debug)[0:100])
        plt.show()
        plt.close()
        
    amplitudes=np.sort(audio_moving)
    if debug:
        plt.hist(amplitudes, bins=100)
        plt.title('Amplitude distribution')
        plt.show()
        plt.close()
    n_amp=audio_moving.shape[0]
    cutoff=amplitudes[int(n_amp*threshold)]
    if debug:
        print('cutoff', cutoff)
    silence_mask=audio_moving < cutoff
    groups = [[i for i, _ in group] for key, group in groupby(enumerate(silence_mask), key=itemgetter(1)) if key]
    boundaries=[(x[0],x[-1]) for x in groups]
    silences=[(x,y) for x,y in boundaries if y-x > min_gap]
    if not silences:
        print('no silence')
        display(Audio(sound, rate=C.sample_rate))
        return (sound, None)
    longest_silence=list(sorted([(y-x,(x,y)) for x,y in silences]))[-1][1]
    start,end = longest_silence
    midpoint = start + (end-start)//2
    split = [sound[0:midpoint], sound[midpoint:]]
    split = [aggressive_clip_ends(clip, C.sample_rate, cutoff) for clip in split]
    if debug:
        print('#silences', len(silences))
        plt.figure(figsize=(50,8))
        plt.plot(sound);
        plt.xlabel('samples')
        plt.ylabel('amplitude');
        for (x1,x2) in silences:
            plt.plot([x1,x2],[0,0],linewidth=10,color='green')
        plt.plot([start, end], [0,0],linewidth=10, color='red')
        plt.title('sound and detected silence')
        plt.show()
        plt.close()
        print('midpoint', midpoint)
        show_sound('LEFT', split[0], C.sample_rate)
        show_sound('RIGHT', split[1], C.sample_rate)
    return split
