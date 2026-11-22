'''
Combines the provided feature directories into one. The resulting directoy contains x and y pairs that can be ingested by a model. The shape of x is padded artificially so that
All x tensors will have the same shape.
'''
import argparse
import torch
from pathlib import Path
import pandas as pd
import os
from Pad import zeropad
max_sequence_length = 6
DEBUG = True

def combine(table, target, *sources):
	'''
	uses the index in the table to find each word in sources and combine it, then saves the result to target.
	'''
	table = pd.read_csv(table)
	table = table.set_index('Unnamed: 0')
	cross_feature_max = getmaxfeaturedimension(sources)
	for i in table.index:
		features = []
		true_nseq = 0
		for k, source in enumerate(sources):
			feature = sorted(list(source.glob(f'{i}_*')))
			feature = [torch.load(f) for f in feature]
			if type(feature[0]) == float:
				feature = [torch.tensor(f) for f in feature]
			padded = [zeropad(f, cross_feature_max) for f in feature]
			padded = [feature_encoder(k + 1,cross_feature_max) + f for f in padded]
			true_nseq = len(padded)
			# ensure that all x tensors are the same shape by adding artificial data which will be masked
			while(len(padded) < max_sequence_length):
				padded.append(torch.zeros(cross_feature_max))
			features.append(padded)
		word = []
		for j in range(max_sequence_length):
			token = [features[k][j] for k in range(len(features))]
			word.append(torch.stack(token))
		y = binarize(table['stress'][i] - 1, true_nseq)
		y = torch.stack([binarize(int(n), 2) for n in y])
		x = torch.stack(word)

		if DEBUG:
			print(x.shape)
			print(y.shape)

		output = Path(f'{target}/{i}')
		output.mkdir()
		torch.save(x, os.path.join(output, Path('x.pt')))
		torch.save(y, os.path.join(output, Path('y.pt')))

def feature_encoder(f: int, dim: int):
	'''
	Takes the id of the feature and applies the vaswani et al function to it. The reasoning for this is that feature vectors with the same
	value should not be identical from the  point of view of the model, adding a feature encoding allows the model to distinguish individual
	features by their unique id.
	'''
	vector = torch.empty((dim,))
	for d in range(len(vector)):
		if d % 2 == 0:
			vector[d] = torch.sin( torch.tensor(f / (10000 ** (d / dim))))
		else:
			vector[d] = torch.cos(torch.tensor(f / (10000 ** (d / dim))))
	return vector


def getmaxfeaturedimension(features: tuple):
	'''
	in order to use feature vectors in the attention network, they should all be padded to the same size. It is assumed that each director has a contant length of vectors
	across all datapoints since the features were generated on a padded audio dataset. This function picks one random file from each folder and returns the maximum vector size
	across all of the folders.
	'''
	max = 0
	for folder in features:
		if DEBUG:
			print("finding maximum", folder)
		file = next(folder.glob('*'))
		tensor = torch.load(file)
		if type(tensor) == float:
			continue
		if DEBUG:
			print(file,'size', len(tensor))
		if len(tensor) > max:
			max = len(tensor)
	if DEBUG:
		print('max feature', max)
	return max

def binarize(i: int, n: int):
	'''
	returns a list of length n, with all zeros except at position i
	'''
	if DEBUG:
		print('binarizing', i, 'of', n, '\n')
	output = torch.zeros((n,))
	output[ i ] = 1
	return output

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("table")
	parser.add_argument("target")
	parser.add_argument("sources", nargs='*')
	args = parser.parse_args()
	combine(Path(args.table), Path(args.target), *[Path(i) for i in args.sources])
