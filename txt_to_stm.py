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

def optional_deletable_space(word):
	if word == "<hes>" or word == "<foreign>" or re.match('^\-.*', word) or re.match('.*\-$', word):
		return "(%s)" % word
	elif re.match('\*.*\*',word):
		return "(%s)" % word.strip('*')
	elif "_" in word and word != "_":
		return re.sub("[\_]"," ",word)
	elif re.match('\/.*\/', word):
		return "%s" % word.strip('/')
	else:
		return word

def tokenize_NIST(label):

	words = label.split(" ")

	#1. delete some separate tags
	words_deleted = list(filter(lambda x: x not in ["(())",
							"<no-speech>",
							"~",
							"<sta>",
							"<lipsmack>", "<breath>", "<cough>", "<laugh>", "<click>", "<ring>", "<dtmf>", "<int>",
							"<male-to-female>", "<female-to-male>"],
				    words))

	#2. convert optional deletable tags and delete remaining tags

	words_tokenized = list(map(optional_deletable_space, words_deleted))

	if len(words_tokenized) > 0:
		#3. replace with the string "IGNORE_TIME_SEGMENT_IN_SCORING" when <overlap> or <prompt> appears.
		if "<overlap>" in words_tokenized or "<prompt>" in words_tokenized:
			label_tokenized = "IGNORE_TIME_SEGMENT_IN_SCORING"
		else:
			label_tokenized = " ".join(words_tokenized)
	else:
		label_tokenized = ""

	label_final = re.sub('\.(?!\d)|\,|\?', '', label_tokenized)
	label_final = label_final.split(' ')
	return label_final

def tokenize(label, no_Q = False):
        if no_Q:
            label = label.replace('Q','')
            label = label.replace('V','')
            label = label.replace('cut','')
        if not label:
            return ''
        try:
                label = ' '.join([x for x in label.split(' ') if len(x) and x[0] not in ('(', '<', '~')])
        except:
                print("label", label)
                quit()
        label = label.replace('*',' ').replace('-', ' ').replace('_', ' ').replace('_', ' ')                 
        words_tokenized = label.split(" ")
        if len(words_tokenized) > 0:
            label_tokenized = " ".join(words_tokenized)
        else:
            label_tokenized = ""
        label_final = re.sub('\.(?!\d)|\,|\?', '', label_tokenized)
        return label_final

def txt_to_stm(df, filename, channel, no_Q = False):

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
			tokenized_text = tokenize(df_new[df_new['time'] == v]['label'].item(), no_Q)
		else:
			tokenized_text = tokenize(str(df_new[df_new['time'] == v]['label'].item()), no_Q)

		#4. if normalized transcript is empty, speakerid will be added interSeg suffix
		if tokenized_text == "":
			speaker_name_new = "{}_{}interSeg".format(filename, channel_name)
			record = [filename, channel_name, speaker_name_new, v, w]

		else:
			record = [filename, channel_name,
                                  "{}_{}".format(filename, channel_name),
                                  v, w, tokenized_text.lower()]

		records.append(record)

	return records

