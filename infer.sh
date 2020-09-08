export PYTHONPATH=/home/catskills/Desktop/openasr20/end2end_asr_pytorch

python end2end_asr_pytorch/infer.py \
	--continue-from save/${language}_end2end_asr_pytorch_drop0.1_cnn_batch12_4_vgg_layer4/best_model.th \
	--test-manifest-list analysis/${language}/DEV_${language}_split.csv \
	--batch-size ${batch_size} \
	--output analysis/${language}/RESULT_${language}.txt \
	--cuda \
	--verbose
