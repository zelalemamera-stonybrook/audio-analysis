'''
The following python program normalizes the effect of natural pitch differences among speakers. Since each individual speaker has natural variation in pitch, the utternce's natural pitch value is not
directly comparible to that of another utterance with a lower natural pitch. The information encoded then measures the deviation of each syllable from a central value within the word.
'''

import pandas as pd
from pathlib import Path
import os
import torch
from Normalize import collect_statistics

def normalize_speakers(source, table, target):
	'''
	computes the average value of vectors over syllables represents each syllable as its distance from that average.
	'''
	print('normalizing speakers for ', source)
	dim, size = collect_statistics(source)
	for i in table.index:
		words = list(source.glob(f'{i}_*.pt'))
		total = torch.zeros((dim,))
		vecs = []
		for word in words:
			vecs.append(torch.load(word))
			if type(vecs[-1]) == float:
				print(vecs[-1])
				vecs[-1] = torch.tensor([vecs[-1]])
			total+= vecs[-1]
		average = total / len(words)
		print('average value found across', len(words), 'syllables', average)
		for word, vec in zip(words, vecs):
			name = os.path.split(word)[-1]
			torch.save(vec - average, os.path.join(target, name))


if __name__ == '__main__':
	import sys
	source = sys.argv[1]
	table = sys.argv[2]
	target = sys.argv[3]
	normalize_speakers(Path(source), pd.read_csv(table).set_index('Unnamed: 0'), Path(target))
