'''
The following program normalizes the input directory. The directory is assumed to be populated with n dimensional vectors. This aligorithm
normalizes along each individual dimension.
'''

import argparse
from pathlib import Path
import torch
import os
DEBUG = True

def normalize(source: str, target: str):
	'''
	computes the center of the data for every dimension, then subtracts each dimension from it. Then divides it by the standard deviation. Finally, a feature id
	is provided which is added to the vector as an encoding.
	'''
	if DEBUG:
		print("normalizing", source)
	dim, size = collect_statistics(source)
	total = torch.zeros((dim,))
	for file in source.iterdir():
		vector = torch.load(file)
		if type(vector) == float:
			vector = torch.tensor([vector])
		total += vector
	mean = total / size
	std = compute_deviation(source, mean, dim, size)
	if DEBUG:
		print('mean is', mean)
		print('std:', std)
	for file in source.iterdir():
		vector = torch.load(file)
		if type(vector) == float:
			vector = torch.tensor([vector])
		normal = torch.nan_to_num(torch.div((vector - mean), std))
		if DEBUG:
			print("normalized", normal)
		filename = os.path.split(file)[-1]
		torch.save(normal, os.path.join(target, filename))


def compute_deviation(source: str, mean: list, dim: int, size: int):
	'''
	Iterates through the soruce directory, computing the expected value of the difference squared from the mean. The root of the result is returned.
	'''
	variance = torch.empty((dim,))
	for file in source.iterdir():
		vector = torch.load(file)
		if type(vector) == float:
			vector = torch.tensor([vector])
		variance += (vector - mean) ** 2
	variance /= size
	return variance ** 0.5

def collect_statistics(source: str):
	'''
	returns the dimension of each vector and the total size of the source directory
	'''
	file = next(source.glob('*'))
	vector = torch.load(file)
	if type(vector) == float:
		vector = torch.tensor([vector])
	dim = len(vector)
	size = len(list(source.glob('*')))
	return dim, size



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("source")
	parser.add_argument("target")
	args = parser.parse_args()
	normalize(Path(args.source), Path(args.target))
