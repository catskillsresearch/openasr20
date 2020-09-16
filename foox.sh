export PYTHONPATH=/home/catskills/Desktop/openasr20/end2end_asr_pytorch

python mytrain.py 	\
	--train-manifest-list train.csv 	\
	--cuda 	\
	--batch-size 6 	\
	--labels-path analysis/amharic/amharic_characters.json  	\
	--lr 1e-4 	\
	--name amharic_4000_end2end_asr_pytorch_drop0.1_cnn_batch12_4_vgg_layer4 	\
	--sample-rate 4000 	\
	--save-folder save 	\
	--epochs 2 	\
	--save-every 1 	\
	--feat_extractor vgg_cnn 	\
	--dropout 0.1 	\
	--num-layers 4 	\
	--num-heads 8 	\
	--dim-model 512 	\
	--dim-key 64 	\
	--dim-value 64 	\
	--dim-input 161 	\
	--dim-inner 2048 	\
	--dim-emb 512 	\
	--shuffle 	\
	--min-lr 1e-6 	\
	--k-lr 1       

