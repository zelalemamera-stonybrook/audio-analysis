'''
Takes a sentence and breaks each word down into syllables, then saves the result to the target directories.
'''
import torch
import torchaudio
import torchcodec
from pathlib import Path
from syllabifier import syllabify
from ExtractOnsets import get_interval, extract_phones

def syllabify_sentence(audiosource, phonemes, words, target):
	'''
	For each word tensor in source, uses the interval in words and the sequence of phones in phonemes to extract its phonetic sequence and uses a syllabifier to break
	it up into syllables. Each syllable sequence is saved to target as a tensor.
	'''
	language = generate_language()
	wordstream = open(words, mode='r')
	audiostream, samplerate = torchaudio.load(audiosource)
	audiostream = audiostream.reshape(-1)
	for i, line in enumerate(wordstream):
		interval = get_interval(line)
		phones = extract_phones(interval[0], interval[1], phonemes)
		print('word:', phones)
		syllabification = syllabify(language, " ".join(phones))
		print('syllabification obtained:', syllabification)
		syllabification = generate_syllables(syllabification, interval, phonemes)
		print(syllabification)
		breakup_word(audiostream, samplerate, syllabification, f'{target}/{i}')

def generate_syllables(syllabification, interval, phonemes):
	'''
	returns the time interval of each syllable in syllabification. First extracts all of the timestamps for the phonemes between the start and end of the word, then makes an alignment between this and the
	syllable list in syllabification to get the start and end timestamps of each syllable.
	'''
	all_intervals = get_all_intervals(interval[0], interval[1], phonemes)
	syllables = []
	i = 0
	for stress, onset, center, coda in syllabification:
		onset+=center
		onset+=coda
		n = len(onset)
		candidate = all_intervals[i: i + n]
		x, y = candidate[0][0], candidate[-1][-1]
		syllables.append((x,y))
		i+=n
	return syllables

def get_all_intervals(x, y, phonemes):
	'''
	Gets all of the integer pairs starting from x and ending with y for each line in phonemes.
	'''
	instream = open(phonemes, mode = 'r')
	intervals = []
	for line in instream:
		interval = get_interval(line)
		if  x <= interval[0] and interval[1] <= y:
			intervals.append((interval[0], interval[1]))
	return intervals

def breakup_word(audiostream, samplerate, syllabification, target):
	'''
	Extracts the subsequence of the word tensor that is provided by the syllabification, then saves the result as tensors to target.
	'''
	i = 0
	print('breaking up', audiostream.shape)
	for x, y in syllabification:
		print(f'{target}/{i}.wav')
		torchaudio.save(f'{target}/{i}.wav',audiostream[x:y], samplerate)
		i+=1
		print('\n')


def generate_language():
	'''
	The syllabifier wants to know the set of characters used for this langauge and the symbols over those characters which
	represent consonants, vowels, and valid onsets.
	'''
	vowels = ['iy','ih','eh','ey','ae','aa','aw','ay','ah','ao','oy','ow','uh','uw','ux','er','ax','ix','axr','ax-h']
	consonants = ['b','d','g','p','t','k','dx','q','jh','ch','s','sh','z','zh','f','th','v','dh','m','n','ng','em','en','eng','nx','l','r','w','y','hh','hv','el','pau','epi','h#','bcl','dcl','gcl','pcl','tck','kcl','tcl']
	onsets = get_onsets()
	return {'consonants':consonants, 'vowels':vowels, 'onsets':onsets}

def get_onsets():
	'''
	At this stage, onsets are assumed to be generated (word initial only) for TIMIT, so can be loaded from a text file located at data_english/onsets.txt
	'''
	inputstream = open('data_english/onsets.txt', mode = 'r')
	onsets = []
	for line in inputstream:
		onsets.append(line.strip())
	return list(set(onsets))

def extract_int(string):
	'''
	pulls out and returns the integer in the string
	'''
	digits = ['0','1','2','3','4','5','6','7','8','9']
	i = 0
	string = list(str(string))
	string.append('\0')
	c = string[i]
	i+=1
	buffer = []
	integers = []
	while(c != '\0'):
		if c in digits:
			while(c in digits):
				buffer.append(c)
				c = string[i]
				i+=1
			integers.append(int(''.join(buffer)))
			buffer = []
		else:
			c = string[i]
			i+=1
	return integers[-1]


if __name__ == '__main__':
	import sys
	audiosource = sys.argv[1]
	phonemes = sys.argv[2]
	words = sys.argv[3]
	target = sys.argv[4]
	syllabify_sentence(Path(audiosource), Path(phonemes), Path(words), Path(target))
