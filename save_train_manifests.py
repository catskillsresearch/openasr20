#!/usr/bin/env python
# coding: utf-8

import  random, json
from tqdm.auto import tqdm
from RecordingCorpus import RecordingCorpus
from SplitCorpus import SplitCorpus

def save_train_manifests(C, pool):
    audio_split_dir=f'{C.build_dir}/audio_split'
    recordings = RecordingCorpus(C, pool)
    splits=SplitCorpus.transcript_split(C, recordings)
    random.shuffle(splits.artifacts)
    n_samples=len(splits.artifacts)
    n_train = int(0.9*n_samples)
    samples=splits.artifacts
    train_samples=samples[0:n_train]
    test_samples=samples[n_train:]
    for (case, S) in [('train', train_samples), ('test', test_samples)]:
        manifest_fn=f'{C.build_dir}/{case}_manifest.json'
        with open(manifest_fn, 'w', encoding='utf-8') as f_manifest:
            for sample in tqdm(S):
                (_,root,(start,end))=sample.key
                audio = sample.source.value
                duration = sample.source.n_seconds
                transcript = sample.target.value
                audio_path=f'{audio_split_dir}/{root}_{start}_{end}.wav'
                metadata = {
                        "audio_filepath": audio_path,
                        "duration": duration,
                        "text": transcript
                    }
                json.dump(metadata, f_manifest)
                f_manifest.write('\n')
        print('saved', manifest_fn)

if __name__ == '__main__':
    import sys
    from Cfg import Cfg
    from multiprocessing import Pool
    language = sys.argv[1]
    C = Cfg('NIST', 16000, language)
    with Pool(16) as pool:
        save_train_manifests(C, pool)
