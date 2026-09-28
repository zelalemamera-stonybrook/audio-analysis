'''
TIMIT comes with a text file containing all of the words used in the transcriptions. It is useful to have a Python dictionary
object that maps the words in this dictionary to their respective phonetic transcriptions. This program creates and stores that dicitionary.
'''
def generate_lexicon_dictionary(text, target):
	'''
	reads in the dicitionary in text and processes it in the appropriate way to produce a Python dictionary and saves it to target.
	'''
	source = open(text, mode='r')
	dictionary = {}
	for line in source:
		line = line.strip()
		if not line or line.startswith(";"):
			continue
		start = line.find("/")
		end = line.rfind("/")
		if start == -1 or end == -1 or start == end:
			continue
		word = line[:start].strip()
		if word.endswith("."):
			word = word[:-1]
		transcription = line[start + 1:end].strip()
		phonemes = transcription.split()
		dictionary[word] = phonemes
	write_dictionary(dictionary, target)
	return dictionary

def write_dictionary(dictionary, target):
	'''
	Writes the Python dictionary to the target file.
	'''
	outputstream = open(target, mode = 'w')
	for key, value in dictionary.items():
		line = f'{key}\t{" ".join(value)}\n'
		outputstream.write(line)


if __name__ == '__main__':
	import sys
	text = sys.argv[1]
	target = sys.argv[2]
	generate_lexicon_dictionary(text, target)
