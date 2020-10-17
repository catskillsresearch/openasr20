#!/usr/bin/bash -x

export runid=115
export language=amharic
export recording=BABEL_OP3_307_17881_20140721_204147_inLine
export ref_file=NIST/openasr20_${language}/dev/transcription_stm/${recording}.stm
export ctm_file=ship/${language}/${runid}/${recording}.ctm
mkdir -p scoring
./SCTK/bin/sclite -r $ref_file stm -h $ctm_file ctm -F -D -O scoring -o sum rsum pralign prf -e utf-8
