#!/usr/bin/bash -x

declare -a languages=("cantonese" "javanese" "mongolian" "somali"
		"vietnamese" "guarani" "kurmanji-kurdish" "pashto" "tamil")

for i in "${languages[@]}"
do
   language.sh $i 100 7 0 1 002
done
