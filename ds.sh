#!/usr/bin/bash -x

export ref_file=NIST/openasr20_amharic/dev/transcription_stm/BABEL_OP3_307_17881_20140721_204147_inLine.stm
export ctm_file=ship/amharic/115/BABEL_OP3_307_17881_20140721_204147_inLine.ctm
mkdir -p scoring
echo ./SCTK/bin/sclite -r $ref_file stm -h $ctm_file ctm -F -D -O scoring/ -o sum rsum pralign prf -e utf-8
