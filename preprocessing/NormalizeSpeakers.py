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
	print("statistics", dim, size)
	for i in table.index:
		words = list(source.glob(f'{i}_*.pt'))
		vecs = [torch.load(i) for i in words]
		if type(vecs[0]) == float:
			vecs = [torch.tensor([i]) for i in vecs]
		vec = torch.cat(vecs)
		avg = torch.mean(vec)
		std = torch.std(vec)
		for word, vec in zip(words, vecs):
			name = os.path.split(word)[-1]
			torch.save((vec - avg) / std, os.path.join(target, name))


if __name__ == '__main__':
	import sys
	source = sys.argv[1]
	table = sys.argv[2]
	target = sys.argv[3]
	table = pd.read_csv(table)
	if "Unnamed: 0" in table.columns:
		table = table.set_index("Unnamed: 0")
	normalize_speakers(Path(source), table, Path(target))
