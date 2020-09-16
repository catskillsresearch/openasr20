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
