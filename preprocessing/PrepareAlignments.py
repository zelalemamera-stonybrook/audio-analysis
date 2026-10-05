'''
The following program prepares audio files for forced alignment. This is the first step of the data processing pipeline that feeds into the StressClassifier neural network.
After this program runs, the relelant audio and text directories have been generated for the alignment.
We use montreal forced aligner to obtain the aligned corpus and proceed to the next step of the pipeline.
The user is ultimately expected to generate the appropriate syllabifications via the Praat software. All of the data necessary to do this will be generated.
'''
import torch
import torchaudio
import torchcodec
from torch import Tensor
from pathlib import Path
import json
import re
import pandas as pd
from pandas import DataFrame
import shutil
import os
import subprocess
seed = 3529145006359120161

path = Path('data/ipa_to_mfa.json')
model_symbols = {}
with path.open(mode='r') as f:
		model_symbols = json.load(f)



def prepare_alignments():
	'''
	the data is assumed to be written into exactly one table. this table will be read from the data directory.
	columns are processed as follows: the text column will be used to generate an arabic to ipa dictionary, as well as a text directory for the alignment.
	the audio urls column will be checked for validity and used to generate an audio directory.
	all of the audio files needed are assumed to be present somewhere in the current directory under the audio subfolder.
	once the necessary data for alignment has been generated, the program is terminated. The text and audio directories generated from the table should be subsequently
	aligned using Montreal Forced Aligner from the command line.
	'''
	print('reading in table')
	data = pd.read_csv(Path('./data/table.csv'))
	data = data.drop(columns='Unnamed: 0')
	print('sucessfuly read table')
	data = {'data': data}

	generate_text(data)
	generate_audio(data)
	generate_aligner_dictionary(data)
	data['data'].to_csv('data/table.csv')

####################################################
#The following function is NOT used in this program
def split_data(data: dict):
	'''
	80/10/10 split
	'''
	print('splitting data')
	split = {}
	for key, val in data.items():
		train = val.sample(frac = .8, random_state=seed // 2 ** 30)
		remainder = val.drop(list(train.index))
		test = remainder.sample(frac = .5, random_state=seed // 2 ** 30)
		dev = remainder.drop(list(test.index))

		split['train'] = train
		split['test'] = test
		split['dev'] = dev
	return split
#
######################################################

def generate_text(split: dict):
	'''
	each table contains a text column which will be used to align the audio file, this text column is used to populate the text directory of the alignment.
	'''
	print('generating text')
	for key, table in split.items():
		#for batch, table in subdict.items():
		address = f'./data/alignment/text'
		shutil.rmtree(address)
		os.mkdir(address)
		for i in table.index:
			write_text_file(i, table['text'][i])
	print('generated text files')

def write_text_file(i: int, text: str):
	'''
	writes text to the directory data/alignment/text/i.txt
	'''
	path = Path(f'./data/alignment/text/{i}.txt')
	with path.open(mode='w') as f:
		f.write(remove_whitespace(text, i))

def remove_whitespace(text: str, i: int):
	'''
	removes whitespace, additionally, due to the presence of identically spelled words in different rows, this function adds the index of the text to it so that each
	element in the text column is guaranteed to be unique.
	'''
	whitespace = re.compile(r'\s')
	temp =  whitespace.sub("", text)
	temp = f'{temp}{i}'
	return temp

def generate_audio(split: dict):
	'''
	each table contains a column locating an audio file for that word, this is copied from its location and used to populate the audio folder of the alignment
	'''
	print('generating audio files')
	for key, table in split.items():
		shutil.rmtree(f'data/alignment/audio')
		os.mkdir(f'data/alignment/audio')
		for i in table.index:
			copy_audio_file(i, table['audio_urls'][i])
	print('generated audio files')


def copy_audio_file(i: int, address: str):
	'''
	copies the file located at data/audio/address to data/alignment/audio/i.wav
	'''
	address = filter_audio_url(address)
	source = Path(f'data/audio/{address}')
	target = Path(f'data/alignment/audio/{i}.wav')
	print(source)
	print(target)
	waveform, samplerate = torchaudio.load(source)
	padded = pad(waveform)
	torchaudio.save(target, padded, samplerate)

def pad(waveform: Tensor):
	'''
	each audio already has some silence at each end. If this silence interval is too small, the aligner will consider the sound endge as
	an end of word boundary. this is not desirable since the syllabifier assumes that the word is contained as a subinterval of the audio. Alternatively,
	If the silence interval is too large, alignment quality deteriorates. This function first removes all silence from both ends of the sound, then adds a 10k bit silence (approximately 1/4th of a second)
	to both ends of the audio
	'''
	waveform = waveform.reshape(-1)
	left = 0
	right = len(waveform) - 1
	n = right
	while (waveform[n] == 0):
		right -=1
		n -=1
	n = left
	while (waveform[n] == 0):
		left +=1
		n +=1
	waveform = waveform[left:right + 1]
	pad1 = torch.zeros((10000,))
	pad2 = torch.zeros((10000,))
	audio = torch.cat((pad1, waveform, pad2))
	return audio

def filter_audio_url(address: str):
	'''
	returns the hashable location of audio
	'''
	return re.search(r"LL-Q55633582.*?wav", address).group()

def generate_aligner_dictionary(data: dict):
	'''
	montreal requires a dictionary of mappings from the arabic, directly to a phoneme sequence in the vocabulary provided by its acoustic model. currently, there is no such dictionary
	so we will generate one from the data by mapping the arabic text directly to its already provided ipa transcription and then mapping this to the character set of the phonemes in MFA.
	the function that maps the observed chracters in the data to phonemes in MFA has already been defined and stored in the directory.
	'''
	print('generating aligner vocabulary')
	dictionary = {}
	for key, value in data.items():
		for i in value.index:
			dictionary[clean_text(value['text'][i], i)] = map_to_mfa(value['ipa'][i])
	write_model_dictionary(dictionary)

def clean_text(txt: str, i: int):
	'''
	processes the text part of data
	'''
	print('received text', txt)
	val = re.sub(r"\s", "", txt)
	val = f'{val}{i}'
	print('writing ', val, 'in dict')
	return val

def write_model_dictionary(dictionary: dict):
	'''
	writes the model dictionary to data/arabic_mfa.dict
	'''
	path = Path(f'data/arabic_mfa.dict')
	with path.open(mode='w') as f:
		for key, value in dictionary.items():
			line = f'{key}\t{value}\n'
			f.write(line)

def map_to_mfa(ipa: str):
	'''
	the raw ipa strings are not suitable for alignment; any suprasegmentals need to be removed if they are not a part of mfa vocabulary,
	mfa strings are delimited by space, however the ipa provided does not come with space delimitation, this needs to be added with the additional complication that any suprasegmental sequences attatched to an
	ipa character should be considered a part of the same token. after this cleaning is applied, the ipa string is sent to the raw 'ipa to model' symbol function to be transformed into an roughly equivalent
	sequence of mfa phone string, which can be used for alignment directly.
	'''
	print('received ipa', ipa)
	ipa = re.sub(r"\s", "", ipa)
	unwanted = set([' ', ')', '(', '‿', 'ˌ', 'ˈ', '.', '͡'])
	suprasegmentals = set(['ˤ', 'ː'])
	ipa_list = list(ipa)
	cleaned_list = []
	segmented_list = []
	for ipa in ipa_list:
		if ipa not in unwanted:
			cleaned_list.append(ipa)
	print('cleaned version of ipa', cleaned_list)
	for i, ipa in enumerate(cleaned_list):
		if ipa in suprasegmentals:
			segmented_list[-1] = f"{cleaned_list[i - 1]}{ipa}"
		else:
			segmented_list.append(ipa)
	for i, ipa in enumerate(segmented_list):
		print('mapping', segmented_list[i], 'to', model_symbols[ipa])
		segmented_list[i] = model_symbols[ipa]
	return " ".join(segmented_list)



if __name__ == '__main__':
	prepare_alignments()
