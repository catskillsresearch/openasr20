#!/usr/bin/bash -x

export language=amharic
export phase=dev
export root=NIST/openasr20_${language}

mkdir -p ${root}/${phase}/transcription_stm

for txt_file in ${root}/${phase}/transcription/*.txt; do

    python OpenASR_convert_reference_transcript.py -f ${txt_file} -o ${root}/${phase}/transcription_stm

done
