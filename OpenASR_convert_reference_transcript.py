#!/usr/bin/python3
# -*-coding:Utf-8 -*
'''
	Usage:

		python OpenASR_convert_reference_transcript.py \
		-f BABEL_BP_101_10470_20111118_172644_inLine.txt \
		-o tmp
'''

import os, re, shutil
import argparse
import pandas as pd
import csv

def tokenize(label):

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

	return label_final

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
			tokenized_text = tokenize(df_new[df_new['time'] == v]['label'].item())
		else:
			tokenized_text = tokenize(str(df_new[df_new['time'] == v]['label'].item()))

		#4. if normalized transcript is empty, speakerid will be added interSeg suffix
		if tokenized_text == "":
			speaker_name_new = "{}_{}interSeg".format(filename, channel_name)
			record = [filename, channel_name, speaker_name_new, v, w]

		else:
			record = [filename, channel_name, "{}_{}".format(filename, channel_name), v, w, tokenized_text]

		records.append(record)

	return records

def convert_from_transcript(args):

	file = "_".join(os.path.basename(args.transcript_file).split("_")[:-1])
	channel = os.path.basename(args.transcript_file).split("_")[-1].split(".")[-2]

	print("Converting {} transcript into stm format".format(args.transcript_file))
	transcript_df = pd.read_csv(args.transcript_file, sep = "\n", header = None, names = ["content"])
	result = txt_to_stm(transcript_df, file, channel)

	transcript_stm = "{}.stm".format(os.path.basename(args.transcript_file).split(".")[0])

	if args.output_directory:
		output_file = os.path.join(args.output_directory, transcript_stm)
		if os.path.exists(output_file):
			os.remove(output_file)
		else:			
			if os.path.exists(args.output_directory):
				pass
			else:
				os.mkdir(args.output_directory)
	else:
		current_directory = os.getcwd()
		output_file = os.path.join(current_directory, transcript_stm)
		if os.path.exists(output_file):
			os.remove(output_file)

	with open(output_file, "w") as output:
		writer = csv.writer(output, delimiter = '	')
		writer.writerows(result)
	output.close()

def main():
    parser = argparse.ArgumentParser(description='Convert transcript into stm format')
    parser.add_argument('-f','--transcript_file', type=str, required=True, help='Input file containing the transcript in dev/train data')
    parser.add_argument('-o','--output_directory', type=str, help='Output directory containing the transcript in STM format')

    parser.set_defaults(func=convert_from_transcript)

    args = parser.parse_args()
    if hasattr(args, 'func') and args.func:
        args.func(args)
    else:
        parser.print_help()

    args = parser.parse_args()

if __name__ == '__main__':
    main()




