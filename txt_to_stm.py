#!/usr/bin/python3
# -*-coding:Utf-8 -*
'''
	Usage:

		python OpenASR_convert_reference_transcript.py \
		-f BABEL_BP_101_10470_20111118_172644_inLine.txt \
		-o tmp
'''

import re
import pandas as pd

def txt_to_stm(df, filename, channel):

	time = []
	label = []

	i, j = 0, 1

	while (i < len(df.content) + 1) and (j < len(df.content) + 1):
		try:
			time.append(df.content[i])
			label.append(df.content[j])
		except:
			pass
		i += 2
		j += 2

	time = [re.sub(r"[\[\]]", "", a) for a in time]

	dict_new = {"time" : time, "label" : label}
	df_new = pd.DataFrame({key:pd.Series(value) for key, value in dict_new.items()})

	records = []
	channel_name = 1 if channel == "inLine" else 2

	for v, w in zip(df_new["time"][:-1], df_new["time"][1:]):

		if isinstance(df_new[df_new['time'] == v]['label'].item(),str):
			tokenized_text = df_new[df_new['time'] == v]['label'].item()
		else:
			tokenized_text = str(df_new[df_new['time'] == v]['label'].item())

		if tokenized_text == '<no-speech>':
                        tokenized_text = ''

		#4. if normalized transcript is empty, speakerid will be added interSeg suffix
		if tokenized_text == "":
			speaker_name_new = "{}_{}interSeg".format(filename, channel_name)
			record = [filename, channel_name, speaker_name_new, v, w]

		else:
			record = [filename, channel_name, "{}_{}".format(filename, channel_name), v, w, tokenized_text]

		records.append(record)

	return records

