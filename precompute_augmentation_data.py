from glob import glob
import os, json
import tqdm
from precompute_augmentation_data_split import precompute_augmentation_data_split
from multiprocessing import Pool
    
def precompute_augmentation_data(language, sample_rate):
    augment_dir=f'NIST/openasr20_{language}/build/audio_augment'
    os.system(f'mkdir -p {augment_dir}')
    audio_fns=glob(f'NIST/openasr20_{language}/build/audio_split/*.wav')
    trans_fns=[audio_fn.replace('.wav','.txt').replace('audio_', 'transcription_') for audio_fn in audio_fns]
    tasks = [(audio_fns[i], trans_fns[i], sample_rate) for i in range(len(audio_fns))]
    with Pool(16) as pool:
        mfests = list(tqdm.tqdm(pool.imap(precompute_augmentation_data_split, tasks), total=len(tasks)))
    manifest=[y for x in mfests for y in x]
    manifest_fn=f'NIST/openasr20_{language}/build/all_manifest.json'
    with open(manifest_fn, 'w', encoding='utf-8') as f_manifest:
        for metadata in manifest:
            json.dump(metadata, f_manifest)
            f_manifest.write('\n')
    print('saved', manifest_fn)

if __name__=="__main__":
    language='vietnamese'
    sample_rate=16000
    precompute_augmentation_data(language, sample_rate)
