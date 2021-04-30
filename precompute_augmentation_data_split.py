import json
import soundfile as sf
from nemo.collections.asr.parts import segment
from Disturb import Disturb

def precompute_augmentation_data_split(task):
    (audio_fn, trans_fn, sample_rate) = task    
    disturb=Disturb(sample_rate)
    sample_segment = segment.AudioSegment.from_file(audio_fn, target_sr=sample_rate)
    duration=sample_segment.samples.shape[0]/sample_rate
    with open(trans_fn, 'r', encoding='utf-8') as f: gold = f.read()
    manifest = [{"audio_filepath": audio_fn, "duration": duration, "text": gold}]
    for i in range(10):
        disturbed=disturb(sample_segment)
        dist_duration=disturbed.samples.shape[0]/sample_rate
        dist_audio_fn=audio_fn.replace('audio_split', 'audio_augment').replace('.wav', f'_{i+1:02d}.wav')
        sf.write(dist_audio_fn, disturbed.samples, sample_rate)
        manifest.append({"audio_filepath": dist_audio_fn, "duration": dist_duration, "text": gold})
    return manifest
