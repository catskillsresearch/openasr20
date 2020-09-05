export language=javanese
python transcript_to_split_BUILD_wavs.py
python transcript_to_grapheme_dictionary.py
python transcript_to_training_file.py
train.sh
python split_DEV_audio.py
python make_DEV_infer_csv.py
infer.sh
python trim_repeats.py
python package_DEV.py
