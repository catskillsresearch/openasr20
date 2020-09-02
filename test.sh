export PYTHONPATH=/home/catskills/Desktop/openasr20/end2end_asr_pytorch

python end2end_asr_pytorch/test.py \
	--continue-from save/amharic_end2end_asr_pytorch_drop0.1_cnn_batch12_4_vgg_layer4/best_model.th \
	--test-manifest-list am_test.csv \
	--cuda

