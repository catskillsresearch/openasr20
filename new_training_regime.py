# coding: utf-8
import pandas as pd
from glob import glob
import sys os
from jiwer import wer
import librosa
from itertools import groupby
from operator import itemgetter
import noisereduce as nr
from unidecode import unidecode
from normalize import normalize
recording='BABEL_OP3_307_82140_20140513_191321_inLine'
gold_fns=list(sorted(glob(f'../../NIST/openasr20_amharic/build/transcription_split/{recording}*.txt')))
goldrows=[]
for fn in gold_fns:
    with open(fn, 'r', encoding='utf-8') as f:
        gold=f.read()
    fsplit=os.path.basename(fn)[0:-4].split('_')
    start,end=fsplit[-2:]
    goldrows.append((float(start), float(end), gold))
df_ref=pd.DataFrame(goldrows, columns=['start', 'end', 'translation'])
df_ref['state']='reference'
df_inf=pd.read_csv(f'build_inference/{recording}.stm',delimiter='\t', names=['recording', 'speaker', 'start', 'end', 'translation'], usecols=[2,3,4])
df_inf['state']='inference'
df=df_ref.merge(df_inf, on='start', how='outer', suffixes=['_ref', '_inf']).sort_values(by='start').dropna(subset=['translation_inf'])
df['match']=df.translation_ref==df.translation_inf
df['wer']=df.apply(lambda x: my_wer(x.translation_ref, x.translation_inf), axis=1)
df1=df[['start', 'end_inf', 'translation_ref', 'translation_inf', 'match', 'wer']]
gold_tgt_fn=[x for x in gold_fns if '68.365' in x][0]
gold_tgt=open(gold_tgt_fn,'r').read()
gold_src_fn=gold_tgt_fn.replace('transcription','audio').replace('txt', 'wav')
gold_src,sr=librosa.load(gold_src_fn, sr=8000)
gold_src=segment_gold_src[0]
(N, threshold, min_gap)=(100, 0.3, 0.3)
audio_moving=np.convolve(gold_src**2, np.ones((N,))/N, mode='same') 
amplitudes=np.sort(audio_moving)
n_amp=audio_moving.shape[0]
cutoff=amplitudes[int(n_amp*threshold)]
silence_mask=audio_moving < cutoff
sample_rate=8000
groups = [[i for i, _ in group] for key, group in groupby(enumerate(silence_mask), key=itemgetter(1)) if key]
boundaries=[(x[0],x[-1]) for x in groups]
silences=[(x/sample_rate,(y-N)/sample_rate) for x,y in boundaries if y-x > min_gap*sample_rate]
longest_noise=sorted([(y-x,(x,y)) for x,y in silences])[-1][1]
noisy_segment=tuple(int(x*sample_rate) for x in longest_noise)
noisy_part = gold_src[noisy_segment[0]:noisy_segment[1]]
reduced_noise = nr.reduce_noise(audio_clip=gold_src, noise_clip=noisy_part, verbose=True)
normed_reduced_noise=normalize(reduced_noise)
end_time=normed_reduced_noise.shape[0]/sample_rate
speech=[]
if silences[0][0] > 0.0:
    speech.append((0,silences[0][0]))
for i in range(len(silences)-1):
    speech.append((silences[i][1],silences[i+1][0]))
if silences[-1][1] < end_time-min_gap:
    speech.append((silences[-1][1], end_time))
speech_segment_lengths=[y-x for x,y in speech]
speech_segment_starts=np.insert(np.cumsum(speech_segment_lengths)[0:-1], 0, 0)
speech_time=sum(speech_segment_lengths)
gold_tgt_words=gold_tgt.split(' ')
gold_tgt_words_chars=sum([len(x) for x in gold_tgt_words])
speech_time_per_char=speech_time/gold_tgt_words_chars
speech_time_per_word=np.array([len(word)*speech_time_per_char for word in gold_tgt_words])
start_times_per_word=np.insert(np.cumsum(speech_time_per_word)[0:-1], 0, 0)
word_to_segment=[np.argmax(np.where(speech_segment_starts <= x)[0]) for x in start_times_per_word]
segments=list(range(len(speech_segment_starts)))
words_in_segment={i:[] for i in segments}
for word, segment in zip(gold_tgt_words, word_to_segment):
    words_in_segment[segment].append(word)
segment_gold_tgt = [' '.join(words_in_segment[segment]) for segment in segments]
segment_gold_src = [normed_reduced_noise[int(x*sample_rate):int(y*sample_rate)] for x,y in speech]
