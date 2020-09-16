#!/usr/bin/bash -x

# Train a language.  Parameters:
#
# 1. Language name
# 2. # Epochs 
# 3. Batch size
# 4. GPU 0 or 1
# 5. Continue from
# 6. Release number

export language=$1
export final_epoch=$2
export batch_size=$3
export CUDA_VISIBLE_DEVICES=$4
export continue_from=$5
export release_number=$6

export PYTHONPATH=/home/catskills/Desktop/openasr20/end2end_asr_pytorch

export recording=BABEL_OP3_307_82140_20140513_191321_inLine

python OpenASR_convert_reference_transcript.py \
       -f NIST/openasr20_amharic/build/transcription/${recording}.txt \
      -o analysis/${language}/build_reference

python end2end_asr_pytorch/infer.py \
	--continue-from save/${language}_end2end_asr_pytorch_drop0.1_cnn_batch12_4_vgg_layer4/best_model.th \
	--test-manifest-list manifest.csv \
	--batch-size ${batch_size} --output analysis/${language}/TEST_BUILD.txt --cuda --verbose

python package_BUILD_as_stm.py ${language} ${recording} analysis/${language}/TEST_BUILD.txt

#SCTK/bin/sclite -m -r analysis/${language}/build_reference/${recording}.stm stm -h analysis/${language}/build_inference/${recording}.ctm ctm
