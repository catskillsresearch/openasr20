import random, json
from json_lines_load import json_lines_load

def reshuffle_samples(C, train_fraction = 0.8, max_duration = 12.0):
    samples=json_lines_load(f'{C.build_dir}/all_manifest.json')
    random.shuffle(samples)
    n_samples=len(samples)
    n_train = int(train_fraction*n_samples)
    samples = [sample for sample in samples if sample['duration'] <= max_duration]
    train_samples=samples[0:n_train]
    test_samples=samples[n_train:]
    for (case, S) in [('train', train_samples), ('test', test_samples)]:
        manifest_fn=f'{C.build_dir}/{case}_manifest.json'
        with open(manifest_fn, 'w', encoding='utf-8') as f_manifest:
            for metadata in S:
                json.dump(metadata, f_manifest)
                f_manifest.write('\n')
    print(f"{len(train_samples)} train_samples, {len(test_samples)} test_samples, split {len(test_samples)/len(samples)}")
