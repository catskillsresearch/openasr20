#!/usr/bin/sh -x

# Train a language.  Parameters:
#
# 1. Language name
# 2. # Epochs 
# 3. Batch size
# 4. GPU 0 or 1
# 5. Continue from

export language=$1
export n_epochs=$2
export batch_size=$3
export CUDA_VISIBLE_DEVICES=$4
export continue_from=$5
# --continue-from ${model_dir}/best_model.th

if [ 0 -eq 1 ]; then
    python transcript_to_split_BUILD_wavs.py
    python transcript_to_grapheme_dictionary.py
    python transcript_to_training_file.py
    train.sh
    python split_DEV_audio.py
    python make_DEV_infer_csv.py
    infer.sh
fi

python trim_repeats.py
python package_DEV.py
