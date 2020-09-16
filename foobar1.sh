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

#   infer.sh
export ifn="analysis/${language}/RESULT_${language}.txt"
export ofn="analysis/${language}/RESULT_${language}_trimwords.txt"
python trim_repeats_in_words.py ${language} ${ifn} ${ofn}

export tfn="analysis/${language}/RESULT_${language}_trimsentences.txt"
python trim_repeats.py ${language} ${ofn} ${tfn}

export efn="analysis/${language}/RESULT_${language}_errorcorrected.txt"
python asr_seq_to_language_seq.py ${language} ${tfn} ${efn}

python package_DEV.py ${language} ${efn}

