import random, json
from json_lines_load import json_lines_load

def reshuffle_samples(C):
    T=json_lines_load(f'{C.build_dir}/train_manifest.json')
    V=json_lines_load(f'{C.build_dir}/test_manifest.json')
    samples=T+V
    random.shuffle(samples)
    n_samples=len(samples)
    n_train = int(0.9*n_samples)
    train_samples=samples[0:n_train]
    test_samples=samples[n_train:]
    for (case, S) in [('train', train_samples), ('test', test_samples)]:
        manifest_fn=f'{C.build_dir}/{case}_manifest.json'
        with open(manifest_fn, 'w', encoding='utf-8') as f_manifest:
            for metadata in S:
                json.dump(metadata, f_manifest)
                f_manifest.write('\n')
