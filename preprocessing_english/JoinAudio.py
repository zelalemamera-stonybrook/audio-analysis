'''
Takes a directory containing audio vectors and writes them to the target directory as wav files.
'''
from SyllabifySentence import extract_int
import torchaudio
import torch
import torchcodec
from pathlib import Path

def join_audio(source, target, id):
	'''
	Each file in source is a syllable of word given by id. These will be saved to the target as id_n where n is the position of the syllable.
	'''
	source = Path(source)
	syllables = sorted(list(source.glob("*.pt")), key = lambda x : extract_int(x))
	for i, syll in enumerate(syllables):
		vector = torch.load(syll)
		address = f'{target}/{id}_{i}.wav'
		print('saving', address)
		torchaudio.save(address, vector)


if __name__ == '__main__':
	import sys
	source = sys.argv[1]
	target = sys.argv[2]
	id = sys.argv[3]
	join_audio(source, target, id)
