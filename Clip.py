import librosa
import numpy as np
from clip_ends import clip_ends
from collect_false import collect_false

class Clip:

    def __init__(self, sample_rate, audio, clipped, parent_start, parent_end, clipped_size):
        self.sample_rate = sample_rate
        self.audio = audio
        self.clipped = clipped
        self.clipped_size = clipped_size
        self.seconds = self.clipped_size / sample_rate
        self.parent_start = parent_start
        self.parent_end = parent_end

    def split_pass(self, max_duration):
        max_samples=int(max_duration*self.sample_rate)
        min_samples=int(0.2*self.sample_rate)
        T=np.arange(self.audio.shape[0])/self.sample_rate
        S = librosa.feature.melspectrogram(y=self.audio, sr=self.sample_rate, n_mels=64, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        s_dB_mean=np.mean(S_dB,axis=0)
        db_grid = np.linspace(-80, -20, 100)
        succeed = False
        for cutoff in db_grid:
            speech_q=(s_dB_mean>cutoff)
            TSQ=T[-1]*np.arange(len(speech_q))/len(speech_q)
            silences=T[-1]*collect_false(speech_q)/len(speech_q)
            if len(silences) == 0:
                continue
            gaps=np.array(list(reversed(sorted(list(set([round(x,3) for x in np.diff(silences).T[0] if x > 0])))))+[0.001])
            for gap in gaps:
                pauses=[(x,y) for x,y in silences if y-x >= gap and int(y*self.sample_rate) < self.audio.shape[0]]
                cuts=np.array([int(self.sample_rate*(x+y)/2) for x,y in pauses if x != 0.0])
                if len(cuts) == 0:
                    continue
                boundaries=np.hstack([[0],cuts,[self.audio.shape[0]]])
                segments=np.array([(boundaries[i], boundaries[i+1]) for i in range(boundaries.shape[0]-1)])
                clips = []
                for start,end in segments:
                    (sound, sound_clipped) = clip_ends(self.audio[start:end])
                    clp = Clip(self.sample_rate, sound, sound_clipped, start + self.parent_start, end + self.parent_start, sound_clipped.shape[0])
                    clips.append(clp)
                sizes = np.array([clip.seconds for clip in clips])
                max_size = sizes.max()
                applicable_clips = [clip for clip in clips if clip.clipped_size <= max_samples]
                problem_clips = [clip for clip in clips if clip.clipped_size > max_samples]
                if len(applicable_clips) > 0:
                    return [applicable_clips, problem_clips]
