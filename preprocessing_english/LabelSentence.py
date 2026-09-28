'''
Takes a sentence treated as a sequence of word strings and writes the stress information of each syllable in every word to
the target directory.
'''
import torch
from ExtractOnsets import extract_phone
from pathlib import Path

def label_sentence(source, words, dictionary, target, id):
	'''
	For each word string in words, looks up its transcription in dictionary and uses this information to obtain its vowel stress information. This information is saved to
	target.
	'''
	dictionary = extract_dictionary(dictionary)
	wordstream = open(words, mode='r')
	error_list = []
	for i, line in enumerate(wordstream):
		word = extract_phone(line)
		labels = []
		if word not in dictionary.keys():
			print('out of vocabulary:', word)
			error_list.append(word)
			labels = get_edgecases(word, id)
			labels = verify_syllable_count(labels, f'{source}/{i}')
		else:
			phonemes = dictionary[word]
			print('labelling', phonemes)
			labels = get_label(phonemes.split(" "))
			labels = verify_syllable_count(labels, f'{source}/{i}')
		print('label obtained', labels)
		for j, label in enumerate(labels):
			gold = f'{target}/{i}/{j}.pt'
			print('saving label', label, 'to', gold)
			torch.save(torch.tensor([label]), gold)
	print('out of vocabulary', error_list)
	outstream = open('error_list.txt', mode='a')
	for word in error_list:
		outstream.write(word)
		outstream.write('\n')

def verify_syllable_count(labels, source):
	'''
	The number of syllables in the source audio direcory does not always match the number of syllables obtained from the lexicon dictionary. This
	function makes a simplifying assumption and pads the remainder at the end of the label vector.
	'''
	source = Path(source)
	syllables = list(source.glob('*.wav'))
	n = len(syllables)
	m = len(labels)
	if n - m > 0:
		for i in range(n - m):
			labels.append(0)
	if n - m < 0:
		labels = torch.zeros((n,))
		labels[0] = 1
		labels = labels.tolist()
	return labels

def get_edgecases(word, i):
	'''
	TIMIT is missing a few words from the dictionary that occur in the transcriptions. Some of these words have a stress location that depends on their intended meaning,
	for example the word present, so that it is not possible to predict the location of the stress without the context.
	'''
	if word in ['live', 'criss', 'zig', 'zagged', 'use']:
		return [1]
	if word == 'present' and i == 175:
		return [0,1]
	if word == 'present' and (i in [274, 272, 68]):
		return [1,0]
	if word == 'wound' and (i in [237]):
		return [1]



def get_label(phonemes):
	'''
	Given a sequence of phones for this word, returns a sequence of labels for each vowel in the word.
	'''
	vowels = ['iy','ih','eh','ey','ae','aa','aw','ay','ah','ao','oy','ow','uh','uw','ux','er','ax','ix','axr','ax-h']
	labels = []
	for c in phonemes:
		stress, phone = get_stress(c)
		if phone in vowels:
			labels.append(stress)
	return labels

def get_stress(c):
	'''
	Extracts an integer if present at the end of c and returns the result as a pair
	'''
	if c[-1] =='1':
		return 1, c[:-1]
	elif c[-1] == '2':
		return 0, c[:-1]
	else:
		return 0, c


def extract_dictionary(dictionary):
	'''
	The dictionary is assumed to have been generated at the location specified. This function reads each line and builds the appopriate
	Python dict.
	'''
	source = open(dictionary, mode='r')
	output = {}
	for line in source:
		key, value = tokenize_line(line)
		output[key] = value
	return output

def tokenize_line(line):
	'''
	Gets the first string before the tab character and the remaining string after the tab character and before the newline character.
	'''
	key = []
	value = []
	i = 0
	c = line[i]
	i+=1
	while(c != '\t'):
		key.append(c)
		c = line[i]
		i+=1
	c = line[i]
	i+=1
	while(c != '\n'):
		value.append(c)
		c = line[i]
		i+=1
	return ''.join(key), ''.join(value)


if __name__ == '__main__':
	import sys
	source = sys.argv[1]
	sentence = sys.argv[2]
	dictionary = sys.argv[3]
	target = sys.argv[4]
	sentenceid = sys.argv[5]
	label_sentence(source, sentence, dictionary, target, int(sentenceid))

