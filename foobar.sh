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
export start_epoch=$7

/bin/rm -rf runs

((i=${start_epoch}))
((j=${final_epoch}))
until [ $i -gt $j  ]
do
  export n_epochs=$i
  echo ${n_epochs}

  python transcript_to_training_file.py
  train.sh

  ((i=i+10))
done

python split_DEV_audio.py
python make_DEV_infer_csv.py
infer.sh
python trim_repeats.py
python package_DEV.py
