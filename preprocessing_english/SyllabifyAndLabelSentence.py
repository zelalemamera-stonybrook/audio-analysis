'''
Takes a sentence and breaks each word down into syllables, then saves the result to the target directories.
'''
from syllabifier import syllabify
from ExtractOnsets import get_interval, extract_phones

def syllabify_sentence(source, phonemes, words, target):
	'''
	For each word tensor in source, uses the interval in words and the sequence of phones in phonemes to extract its phonetic sequence and uses a syllabifier to break
	it up into syllables. Each syllable sequence is saved to target as a tensor.
	'''
	words = list(source.glob("*.pt"))
	words = sorted(words, key = lambda x : extract_int(x))
	language = generate_language()
	wordstream = open(words, mode='r')
	for i, line in enumerate(wordstream):
		interval = get_interval(line)
		phones = extract_phones(interval[0], interval[1], phonemes)
		syllabification = syllabify(language, phones)
		breakup_word(words[i], syllabification, target)

def breakup_word(word, syllabification, target):
	'''
	Extracts the subsequence of the word tensor that is provided by the syllabification, then saves the result as tensors to target.
	'''

def extract_dictionary(dictionary):
	'''
	Goes through the TIMIT dictionary and makes a python dict object with keys as the words and values as the phoneme strings provided in the file, returns the result.
	'''


def breakup_words(words, syllables, intervals):
	'''
	Breaks each tensor located in words by syllables using the information in syllables and intervals
	'''
def generate_language():
	'''
	The syllabifier wants to know the set of characters used for this langauge and the symbols over those characters which
	represent consonants, vowels, and valid onsets.
	'''
	vowels = ['iy','ih','eh','ey','ae','aa','aw','ay','ah','ao','oy','ow','uh','uw','ux','er','ax','ix','axr','ax-h']
	consonants = ['b','d','g','p','t','k','dx','q','jh','ch','s','sh','z','zh','f','th','v','dh','m','n','ng','em','en','eng','nx','l','r','w','y','hh','hv','el','pau','epi','h#','bcl','dcl','gcl','pcl','tck','kcl','tcl']
	onsets = get_onsets()
	return {'consonants':consonants, 'vowels':vowels, 'onsets':onsets}

def get_onsets()
	'''
	At this stage, onsets are assumed to be generated (word initial only) for TIMIT, so can be loaded from a text file located at data_english/onsets.txt
	'''
	inputstream = open('data_english/onsets.txt', mode = 'r')
	onsets = []
	for line in inputstream:
		onsets.append(line)
	return list(set(onsets))

def extract_int(str):
	'''
	pulls out and returns the integer in the string
	'''
	digits = ['0','1','2','3','4','5','6','7','8','9']
	i = 0
	str = list(str)
	str.append('\0')
	c = str[i]
	i+=1
	buffer = []
	integers = []
	while(c != '\0'):
		if c in digits:
			while(c in digits):
				buffer.append(c)
				c = str[i]
				i+=1
			integers.append(int(''.join(buffer)))
			buffer = []
		else:
			c = str[i]
			i+=1
	return integers[-1]


if __name__ == '__main__':
	import sys
	source = sys.argv[1]
	phonemes = sys.argv[2]
	words = sys.argv[3]
	target = sys.argv[4]
	syllabify_sentence(Path(source), Path(phonemes), Path(words), Path(target))
