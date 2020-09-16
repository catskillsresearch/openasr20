import os, sys


def runtrainer(C, samples, n_epochs, batch_size):

    C.cfrom=f"--continue-from {C.model_dir}/best_model.th" if os.path.exists(C.best_model) else ''

    with open('train.csv', 'w') as f:
        f.write('\n'.join([f"{x},{x.replace(f'/audio_split_{C.sample_rate}/', '/transcription_split/').replace('.wav', '.txt')}"
                           for x in samples]))

    os.system(f"""python mytrain.py \
	--train-manifest-list train.csv \
	--cuda \
	--batch-size {batch_size} \
	--labels-path {C.grapheme_dictionary_fn}  \
	--lr 1e-4 \
	--name {C.model_name} \
	--sample-rate {C.sample_rate} \
	--save-folder save \
	--epochs {n_epochs} \
	--save-every 1 \
	--feat_extractor vgg_cnn \
	--dropout 0.1 \
	--num-layers 4 \
	--num-heads 8 \
	--dim-model 512 \
	--dim-key 64 \
	--dim-value 64 \
	--dim-input 161 \
	--dim-inner 2048 \
	--dim-emb 512 \
	--shuffle \
	--min-lr 1e-6 \
	--k-lr 1 \
	{C.cfrom}""")

if __name__=="__main__":
    from glob import glob
    import librosa
    import numpy as np
    from config import C
    os.system("/bin/rm -rf runs")
    files=glob(f'{C.nr_dir}/*')
    n_samples=len(files)
    samples=list(sorted([(librosa.load(audio_file, sr=C.sample_rate)[0].shape[0], audio_file) for audio_file in files]))

    sieve = 500
    sample_ends=(n_samples*np.arange(sieve)/sieve).astype(int).tolist()[1:]+[n_samples]

    # Hand tune when it breaks
    batch_size = {i: 12 for i in range(len(sample_ends))}

    for i, end in enumerate(sample_ends):
        bs = batch_size[i]
        print("------------------------------------------------")
        print(f"[{i}] batch_size {bs} samples {end}")
        runtrainer(C, [y for x,y in samples[0:end]], max(2,10-i), batch_size[i])
