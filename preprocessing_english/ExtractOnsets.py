'''
Takes a phoneme sequence for a sentence and word boundaries, and uses this to extract all of the word initial onsets from the sentence.
The set of vowel symbols used is assumed to be known.
'''
from pathlib import Path

def extract_onsets(phonemes, words, target):
	'''
	Extracts each word initial onset in phonemes and appends it to target.
	'''
	vowels = ['iy','ih','eh','ey','ae','aa','aw','ay','ah','ao','oy','ow','uh','uw','ux','er','ax','ix','axr','ax-h']
	wordintervals = extract_all_intervals(words)
	outputstream = open(target, mode = 'a')
	for x, y in wordintervals:
		phones = extract_phones(x, y, phonemes)
		onset = find_word_initial_onset(phones, vowels)
		if len(onset) == 0:
			continue
		write_onset(onset, outputstream)

def write_onset(onset, outputstream):
	'''
	Appends the onset to target on a new line and separated by one whitespace.
	'''
	onset = (' '.join(onset)).strip()
	line = f'{onset}\n'
	outputstream.write(line)

def find_word_initial_onset(phones, vowels):
	'''
	given the phone sequence, and a vocabulary of vowels, returns the largest subsequence that precedes the first occurence of a vowel.
	'''
	onset = []
	for phn in phones:
		if phn in vowels:
			return onset
		else:
			onset.append(phn)
	return onset

def extract_all_intervals(words):
	'''
	Returns the integer intervals specified at the beginning of each line in words as a list of pairs.
	'''
	inputstream = open(words, mode = 'r')
	intervals = []
	for line in inputstream:
		intervals.append(get_interval(line))
	return intervals

def get_interval(line):
	'''
	Extracts the integer interval at the beginning of the line and returns it as a pair.
	'''
	digits = ['0','1','2','3','4','5','6','7','8','9']
	line = list(line)
	line.append('\0')
	i = 0
	c = line[i]
	i+=1
	buffer = []
	interval = []
	while(c != '\0'):
		if c in digits:
			while(c in digits and c!='\0'):
				buffer.append(c)
				c = line[i]
				i+=1
			interval.append(int("".join(buffer)))
			buffer = []
		else:
			c = line[i]
			i+=1
	return interval

def extract_phones(x, y, phonemes):
	'''
	Returns the sequence of phone characters that begins at time x and ends at time y.
	'''
	inputstream = open(phonemes, mode = 'r')
	phones = []
	for line in inputstream:
		interval = get_interval(line)
		if interval[0] >= x:
			phones.append(extract_phone(line))
		if interval[1] >= y:
			break
	return phones

def extract_phone(line):
	'''
	Pulls out the character string at the end of the line and returns it.
	'''
	digits = ['0','1','2','3','4','5','6','7','8','9']
	line = list(line)
	for i, c in enumerate(line):
		if c == '\n':
			line[i] = '\0'
	i = 0
	c = line[i]
	i+=1
	buffer = []
	while(c != '\0'):
		if c not in digits and c != ' ':
			while(c not in digits and c != ' ' and c != '\0'):
				buffer.append(c)
				c = line[i]
				i+=1
			return (''.join(buffer)).strip()
		else:
			c = line[i]
			i+=1

	return ''

if __name__ == '__main__':
	import sys
	print(sys.argv)
	extract_onsets(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))

