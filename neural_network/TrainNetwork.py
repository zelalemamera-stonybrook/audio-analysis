'''
This program controls the training of the provided neural network. The SGD algorithm is used.
'''
import importlib
import argparse
from pathlib import Path
import json
import torch
import torchaudio
from torch import Tensor
import os
from Pad import zeropad
import pandas as pd
from pandas import DataFrame
import re
SEED = 3529145006359120161

DEBUG = False

def train(network, source: str, table: str, log: str, batchsize: int, *features):
	'''
	Trains the neural network exactly one epoch. An epoch is defined as one training pass over the input data. The data is batched before being passed to the model. Error is computed per batch
	and backpropagated before loading in the next batch.
	'''
	attention = False
	table = pd.read_csv(table)
	table = table.set_index('Unnamed: 0')

	error_history = []
	optim = torch.optim.SGD(network.parameters(), lr=0.001,  momentum=0.5)
	network.feature_weights = []

	numberofbatches = len(table) // batchsize
	if len(table) % batchsize != 0:
		numberofbatches +=1
	table = groupbatches(table, batchsize)
	maxfeaturedimension = getmaxfeaturedimension(features)
	for i in range(numberofbatches):
		optim.zero_grad()
		error = 0
		x, y = getbatch(i, source, table)
		if attention:
			f = getfeaturebatch(i, features, table, maxfeaturedimension)
		else:
			f = listfull(None, batchsize)
		for vector, feature, gold in zip(x, f, y):
			if DEBUG:
				vectori = torch.stack(vector)
				print('input vector statistics', 'length', vectori.shape, torch.min(vectori).item(), torch.max(vectori).item(), torch.mean(vectori).item(), 'dimensions', torch.count_nonzero(vectori, dim=1).tolist())
				if feature:
					featurei = torch.stack([torch.stack(i) for i in feature])
					print('input feature statistics', featurei.shape, torch.min(featurei).item(), torch.max(featurei).item(), torch.mean(featurei).item())
				print(gold)
			y_hat = network.forward(vector)
			if True:
				print( y_hat, 'gold', gold)
			error += compute_loss(y_hat, gold)
		if True:
			print('batch error', error)
		error_history.append(error.item())
		print('backpropagating the error')
		error.backward()
		print('updating the parameters')
		optim.step()
	writemodel(network, f'{network.name}.json')
	logerrorhistory(error_history, log)

def groupbatches(table: DataFrame, batchsize:int):
	'''
	returns a table that is split up into modulo batchsize batches, if the table is not divisible by batchsize, the last batch is returned as the remainer less than batchsize
	'''
	table = [tup for tup in zip(table.index, table['stress'])]
	n = len(table) // batchsize
	batches = []
	for i in range(n):
		batches.append(table[i*batchsize: i*batchsize +  batchsize])
	if len(table) % batchsize != 0:
		batches.append(table[n * batchsize:])
	if DEBUG:
		print('batch list is', '[', *batches[:3], "...", *batches[-3:], ']')
	return batches


def getbatch(i: int, source: str, table: list):
	'''
	the table is assumed to be split up into modulo batchsize batches. each batch is a collection of file ids that correspond to a word. Each word in turn is represented by
	an unspecfied number of syllables. These syllables are the basic units that are stored in the source, so they must be collected and returned in the same order as specified by the table.
	in addition to the source files, a corresponding list of gold targets is also returned.
	'''
	batch = table[i]
	x = []
	y = []
	for i, j in batch:
		word = []
		files = sorted(list(source.glob(f'{i}_*')))
		extension = getextension(files[0])
		if DEBUG:
			print('word to be read', files)
		for file in files:
			tensor = torch.load(file)
			word.append(tensor)
			if DEBUG:
				print('embedding for', file, word[-1].shape)
		x.append(word)
		y.append(binarize(j-1,len(word)))
		if DEBUG:
			print('gold label for word', y[-1])
	return x, y

def getfeaturebatch(i: int, features: tuple, table: list, max: int):
	'''
	gets the features from batch i of the table. for each word in the batch, features is a list of directories that contain the different representations of that word.
	'''
	global SEED
	batch = table[i]
	output = []
	for i, j in batch:
		wordfeatures = []
		for folder in features:
			word = []
			files = sorted(list(folder.glob(f'{i}_*')))
			for file in files:
				tensor = torch.load(file)
				if type(tensor) == float:
					tensor = torch.tensor([tensor])
					g_cpu = torch.Generator()
					tensor = (torch.nn.init.uniform_(torch.empty((max,)), -1, 1, g_cpu.manual_seed(SEED))) * tensor
				word.append(zeropad(tensor, max))
			wordfeatures.append(word)
		output.append(wordfeatures)
	return output

def getmaxfeaturedimension(features: tuple):
	'''
	in order to use feature vectors in the attention network, they should all be padded to the same size. It is assumed that each directory has a contant length of vectors
	across all datapoints since the features were generated on a padded audio dataset. This function picks one random file from each folder and returns the maximum vector size
	across all of the folders.
	'''
	max = 0
	for folder in features:
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

def getextension(file: str):
	'''
	returns the str following . after the filename, if it exists
	'''
	return re.split(r'\.',str(file))[-1]

def listfull(obj: object, n: int):
	'''
	returns a list of size n full of the object
	'''
	return [obj for i in range(n)]

def writemodel(network, name: str):
	'''
	writes the model's current weights to the directory located at neural_network as name
	'''

	state_dict = network.state_dict(keep_vars = True)
	for key, value in state_dict.items():
		state_dict[key] = value.tolist()
	path = Path(os.path.join(Path('neural_network'), Path(f'{name}')))
	json.dump(state_dict, path.open(mode='w'))

def logerrorhistory(error_history: list, log: str):
	'''
	writes the error history to the target file
	'''
	error_history = torch.tensor(error_history)
	with log.open(mode='a') as file:
		file.write('\n----------------------------------------\n')
		file.write(f'min: {torch.min(error_history).item()} max: {torch.max(error_history).item()} mean: {torch.mean(error_history).item()}')

def analyze_optimstate_dict(state_dict: dict):
	'''
	looks at model's previous gradients for any anomalies.
	'''
	for key, value in state_dict['state'].items():
		gradient = value['momentum_buffer']
		print(key, torch.min(gradient), torch.max(gradient), gradient.shape)

def analyze_state_dict(state_dict: dict):
	'''
	looks at the models parameters for any anomalies.
	'''
	for key, value in state_dict.items():
		print(key, value.shape)



def sample_analysis(N1: int, N: int, input:str, gold:list):
	'''
	randomly samples classes and analyzes the distribution
	'''
	n = torch.randint(N1, N, (1000,))
	d = {1:0, 2:0, 3:0, 4:0}
	k = {2:0, 3:0, 4:0}
	for i in n:
		value = gold[i]
		syll = value[0]
		k[syll] +=1
		location = value[-1]
		for j in range(syll):
			if location[j] == 1:
				d[j+1] +=1
	distribution = []
	distributionw = []
	for key, value in d.items():
		distribution.append(value / 1000)
	for key, value in k.items():
		distributionw.append(value / 1000)
	print('distribution obtained over location classes')
	print(distribution)
	print('distribution obtained over syllable classes')
	print(distributionw)

def analyze_graph(y: Tensor):
	'''
	analyzes the graph function of y
	'''
	if y == None:
		return
	print(y.next_functions)
	for f, k in y.next_functions:
		analyze_graph(f)


def compute_loss(y_hat: Tensor, y: Tensor):
	'''
	measures the norm of distance of the prediction y hat to y and returns the result.
	y_hat shape: (n, 2)
	y shape: (n)
	'''
	if DEBUG:
		print('computing loss from\n', y_hat, y)
	binary_list = []
	for n in y:
		binary_list.append(binarize(int(n), 2))
		if DEBUG:
			print('gold distribution', binary_list[-1])
	y = torch.Tensor(binary_list)
	distance = torch.add(y, y_hat, alpha=-1)
	error = 0.5 * (torch.linalg.vecdot(distance, distance)).sum()
	if DEBUG:
		print('distance computed', error)
	return error

def binarize(i: int, n: int):
	'''
	returns a list of length n, with all zeros except at position i
	'''
	output = torch.zeros((n,))
	output[ i ] = 1
	return output.tolist()

def loadmodel(module: str, reset: bool):
	'''
	loads the model source code and pretrained parameters if any.
	'''

	module = importlib.import_module(module)
	network = None
	if reset:
		network = module.Network()
		if DEBUG:
			print('network is', *[f'{i.shape}\n' for i in network.parameters()])
		return network
	else:
		network = module.Network()
		path = Path(f'neural_network/{network.name}.json')
		model_parameters = json.load(path.open(mode='r'))
		for key, value in model_parameters.items():
			if key == 'name':
				continue
			model_parameters[key] = torch.nn.parameter.Parameter(torch.tensor(value), requires_grad = True)
		network.load_state_dict(model_parameters)
		if DEBUG:
			print('network is', *[f'{i.shape}\n' for i in network.parameters()])
		return network


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('network')
	parser.add_argument('source')
	parser.add_argument('table')
	parser.add_argument('log')
	parser.add_argument('batchsize')
	parser.add_argument('-r',action = 'store_true', help='reset model parameters')
	parser.add_argument('features', nargs='*', default=None, help='provide feature directory sources')
	args = parser.parse_args()
	network = loadmodel(args.network, args.r)
	train(network, Path(args.source), Path(args.table),Path(args.log), int(args.batchsize), *[Path(i) for i in args.features])



