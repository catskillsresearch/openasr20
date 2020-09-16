#!/usr/bin/sh -x

export PYTHONPATH=/home/catskills/Desktop/openasr20/end2end_asr_pytorch

export model_dir=save/${language}_end2end_asr_pytorch_drop0.1_cnn_batch12_4_vgg_layer4

mkdir -p ${model_dir}

if [ $continue_from -eq 1 ]; then
   export cfrom="--continue-from ${model_dir}/best_model.th"
fi

python end2end_asr_pytorch/train.py \
	--train-manifest-list analysis/${language}/${language}_train.csv \
	--valid-manifest-list analysis/${language}/${language}_valid.csv \
	--cuda \
	--batch-size ${batch_size} \
	--labels-path analysis/${language}/${language}_characters.json  \
	--lr 1e-4 \
	--name ${language}_end2end_asr_pytorch_drop0.1_cnn_batch12_4_vgg_layer4 \
	--sample_rate 8000 \
	--save-folder save \
	--epochs ${n_epochs} \
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
	${cfrom}
