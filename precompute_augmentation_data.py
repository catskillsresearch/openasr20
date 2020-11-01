from glob import glob
import os, json
import soundfile as sf
from tqdm.auto import tqdm
from Disturb import Disturb

def precompute_augmentation_data(language, sample_rate):
    disturb=Disturb(sample_rate)
    augment_dir=f'NIST/openasr20_{language}/build/audio_augment'
    os.system(f'mkdir -p {augment_dir}')
    audio_fns=glob(f'NIST/openasr20_{language}/build/audio_split/*.wav')
    manifest=[]
    trans_fns=[audio_fn.replace('.wav','.txt').replace('audio_', 'transcription_') for audio_fn in audio_fns]
    for (audio_fn, trans_fn) in tqdm(zip(audio_fns, trans_fns)):
        sample_segment = segment.AudioSegment.from_file(audio_fn, target_sr=sample_rate)
        duration=sample_segment.samples.shape[0]/sample_rate
        with open(trans_fn, 'r', encoding='utf-8') as f: gold = f.read()
        manifest.append({"audio_filepath": audio_fn, "duration": duration, "text": gold})
        for i in range(10):
            disturbed=disturb(sample_segment)
            dist_duration=disturbed.samples.shape[0]/sample_rate
            dist_audio_fn=audio_fn.replace('audio_split', 'audio_augment').replace('.wav', f'_{i+1:02d}.wav')
            sf.write(dist_audio_fn, disturbed.samples, sample_rate)
            manifest.append({"audio_filepath": dist_audio_fn, "duration": dist_duration, "text": gold})
    manifest_fn=f'NIST/openasr20_{language}/build/all_manifest.json'
    with open(manifest_fn, 'w', encoding='utf-8') as f_manifest:
        for metadata in manifest:
            json.dump(metadata, f_manifest)
            f_manifest.write('\n')
    print('saved', manifest_fn)

if __name__=="__main__":
    language='vietnamese'
    sample_rate=16000
