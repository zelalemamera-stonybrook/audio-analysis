'''
Appends the current word's data to the gold table. The set of infromation collected on this word includes, number of syllables, phonetic transcription,
word id and stress location.
'''
import torch
from pathlib import Path
from LabelSentence import extract_dictionary
from ExtractOnsets import extract_phone
from SyllabifySentence import extract_int

def join_gold(goldsource, wordsource, wordid, sentenceid, dictsource, id, target):
	'''
	Generates a table of the following format:

	text	ipa	syllables	stress
	present	<>	2	2

	this table is saved to the target
	'''
	print("labelling sentence", sentenceid, "word", wordid, "\n")
	labels = sorted(list(Path(goldsource).glob("*.pt")), key = lambda x: extract_int(x))
	word = get_word(wordid, wordsource)
	dictionary = extract_dictionary(dictsource)
	label = 0
	transcription = ''
	if len(labels) == 0:
		raise ValueError("no labels found")
	if word not in dictionary.keys():
		label = get_label(labels)
		transcription = 'unavailable'
	else:
		transcription = dictionary[word]
		label = get_label(labels)
	if Path(target).exists():
		outputstream = open(target, mode = 'a')
		line = f'{word},{transcription},{len(labels)},{label}\n'
		print(line)
		outputstream.write(line)
	else:
		outputstream = open(target, mode = 'a')
		line = 'text,ipa,syllables,stress\n'
		outputstream.write(line)
		print(line)
		line = f'{word},{transcription},{len(labels)},{label}\n'
		print(line)
		outputstream.write(line)

def get_word(id, wordsource):
	'''
	Returns the string for word i in wordsource.
	'''
	inputstream = open(wordsource, mode='r')
	for i, line in enumerate(inputstream):
		if i == id:
			return extract_phone(line)
		else:
			continue

def get_label(labels):
	'''
	returns the integer position where the tensor in labels is not zero
	'''
	tensors = [torch.load(t) for t in labels]
	i = 0
	for t in tensors:
		if t.item() != 0:
			return i + 1
		else:
			i+=1


if __name__ == '__main__':
	import sys
	goldsource = sys.argv[1]
	wordsource = sys.argv[2]
	wordid = sys.argv[3]
	sentenceid = sys.argv[4]
	dictsource = sys.argv[5]
	id = sys.argv[6]
	target = sys.argv[7]
	join_gold(goldsource, wordsource, int(wordid), int(sentenceid), dictsource, int(id), target)
