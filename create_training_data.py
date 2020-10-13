import random, librosa, json
import soundfile as sf
from tqdm.auto import tqdm
from RecordingCorpus import RecordingCorpus
from multiprocessing import Pool
from contextlib import closing
from SplitCorpus import SplitCorpus

def create_training_data():
    with closing(Pool(16)) as pool:
        recordings = RecordingCorpus(C, pool)
    splits=SplitCorpus.transcript_split(C, recordings)
    random.shuffle(splits.artifacts)
    n_samples=len(splits.artifacts)
    n_train = int(0.8*n_samples)
    samples=splits.artifacts
    train_samples=samples[0:n_train]
    test_samples=samples[n_train:]
    audio_split_dir=f'{C.build_dir}/audio_split'
    upsample_rate=params['sample_rate']
    for (case, S) in [('train', train_samples), ('test', test_samples)]:
        manifest_fn=f'{C.build_dir}/{case}_manifest.json'
        with open(manifest_fn, 'w', encoding='utf-8') as f_manifest:
            for sample in tqdm(S):
                (_,root,(start,end))=sample.key
                audio = sample.source.value
                duration = sample.source.n_seconds
                transcript = sample.target.value
                audio_path=f'{audio_split_dir}/{root}_{start}_{end}.wav'
                sf.write(audio_path, audio, C.sample_rate)
                y3, sr3 = librosa.load(audio_path,sr=upsample_rate)
                sf.write(audio_path, y3, upsample_rate)
                metadata = {
                        "audio_filepath": audio_path,
                        "duration": duration,
                        "text": transcript
                    }
                json.dump(metadata, f_manifest)
                f_manifest.write('\n')
