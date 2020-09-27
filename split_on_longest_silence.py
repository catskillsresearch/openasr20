import numpy as np
from IPython.display import Audio

def split_on_longest_silence(sound, sample_rate, _cutoff = 0.0014, debug=None):
    N=100
    min_gap=0.04*sample_rate
    audio_moving=np.convolve(sound**2, np.ones((N,))/N, mode='same') 
    if debug:
        print(f'average energy over {N} sample window')
        plt.figure(figsize=(40,4))
        plt.plot(sound)
        plt.plot(audio_moving);
        plt.title(unidecode(debug))
        plt.show()
    if 0:
        threshold=0.3
        amplitudes=np.sort(audio_moving)
        n_amp=audio_moving.shape[0]
        cutoff=amplitudes[int(n_amp*threshold)]
    else:
        cutoff = _cutoff
    if debug:
        print('cutoff', cutoff)
    silence_mask=audio_moving < cutoff
    groups = [[i for i, _ in group] for key, group in groupby(enumerate(silence_mask), key=itemgetter(1)) if key]
    boundaries=[(x[0],x[-1]) for x in groups]
    silences=[(x,(y-N)) for x,y in boundaries if y-x > min_gap]
    if not silences:
        print('no silence')
        display(Audio(sound, rate=C.sample_rate))
        return (sound, None)
    if debug:
        print('#silences', len(silences))
        plt.figure(figsize=(50,8))
        plt.plot(sound);
        plt.xlabel('seconds')
        plt.ylabel('amplitude');
        for (x1,x2) in silences:
            plt.plot([x1,x2],[0,0],linewidth=5,color='red')
        plt.title('sound and detected silence')
        plt.show()
    longest_silence=list(sorted([(y-x, (x,y)) for x,y in silences]))[-1][1]
    midpoint_of_longest_silence=longest_silence[0]+((longest_silence[1]-longest_silence[0])//2)
    speech=[(0,midpoint_of_longest_silence), (midpoint_of_longest_silence,sound.shape[0])]
    sounds=[clip_ends(sound[a:b], 0.0008) for a,b in speech]
    if debug:
        spliced=np.hstack(sounds)
        plt.figure(figsize=(50,8))
        plt.plot(spliced);
        plt.xlabel('seconds')
        plt.ylabel('amplitude');
        plt.title('spliced')
        print(f'sound: {unidecode(debug)}')
        display(Audio(sound, rate=C.sample_rate))
        print('spliced')
        display(Audio(spliced, rate=C.sample_rate))
        for i, sound in enumerate(sounds):
            print('segment', i)
            display(Audio(sound, rate=C.sample_rate))
            plt.figure(figsize=(50,8))
            plt.plot(sound);
            plt.xlabel('seconds')
            plt.ylabel('amplitude');
            plt.title(f"segment {i}")
        plt.show()
    return sounds
