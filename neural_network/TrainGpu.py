'''
In order to take advantage of the full capabilities of the gpu, we need to batch computations into large chunks that depend only on our available memory.
'''
import json
import importlib
from pathlib import Path
import Dataset
import torch
import torch.nn as nn
import pandas as pd
import TestGpu
import time
DEBUG = False
ELS = 0.10
OUTPUTDIM = 512
WARMUP = 100
MAXNORM = 10

device = torch.device('cpu')
torch.set_default_device(device)
torch.set_default_dtype(torch.float64)

def train(dataloader, network, epochs, parameters=1):
	'''
	Backpropagates error once over the dataset contained in source. The network takes a single step. The batches over which error is computed is determined based on
	the available memory. The algorithm tries to use the biggest batch possible, as this gives the biggest reduction in training time.
	'''
	i = 1
	for n in range(epochs):
		network.train(True)
		for x, y, mask in dataloader:
			#if i >  0:
			#	break
			optimizer = torch.optim.Adam(network.parameters(), betas=(0.9, 0.98), eps= 10 ** -9, lr=1)
			if DEBUG:
				print("input", x.shape)
			print("remaining batches", len(dataloader) - i % len(dataloader))
			print("forwardpass begins")
			start = time.perf_counter()
			y_hat = network(x, mask)
			end = time.perf_counter()
			print("forwardpass complete", end - start, "s")
			if True:
				print("prediction obtained")
				print(y_hat[:3], y_hat.shape)
			y_hat = unmask(y_hat, mask)
			error = error_function(y_hat, y)
			print("backpropagation begins\n")
			start = time.perf_counter()
			error.backward()
			print("backpropagation complete", end - start, "s")
			grad_norm = nn.utils.clip_grad_norm_(network.parameters(), 200)
			print("gradient norm", grad_norm)
			end = time.perf_counter()
			optimizer.step()
			i+=1
		fscore = TestGpu.test(network, 'results/Transformer/hypothesis', 'data/DEV/table.csv', 'results/Transformer', n, 'data/DEV/Praat')
		if fscore:
			if fscore >= 90:
				writemodel(network, f'{network.name}')

def writemodel(network, name):
	'''
	writes the model's current weights to the directory located at neural_network as name
	'''
	start = time.perf_counter()
	print("writing model begins")
	state_dict = network.state_dict(keep_vars = True)
	for key, value in state_dict.items():
		state_dict[key] = value.tolist()
	path = f'neural_network/Model.json'
	json.dump(state_dict, open(path, mode='w'))
	end = time.perf_counter()
	print("writing model complete", end - start, "s")


def unmask_transpose(y_hat_batch, mask):
	'''
	Filters out all but the unmasked elements of y_hat for backpropagation
	'''
	unmasked = [torch.stack([y_hat_batch[j][i] for j in range(mask[i])]) for i in range(len(mask))]
	return unmasked

def unmask(y_hat_batch, mask):
	unmasked = [torch.stack([y_hat_batch[i][j] for j in range(mask[i])]) for i in range(len(mask))]
	return unmasked

def error_function(y_hat_batch, y_batch):
	'''
	measures the norm of distance of the prediction y hat to y for each prediction in the batch.
	y_hat shape: (n, 2)
	y shape: (n, 2)
	'''
	if DEBUG:
		print('computing loss from\n', y_hat_batch[:3],"\n", y_batch[:3])
	error = 0
	for y_hat, y in zip(y_hat_batch, y_batch):
		distance = y_hat - label_smoothing(y, ELS)
		error += 0.5 * (torch.linalg.vecdot(distance, distance)).sum()
	print('batch error', error)
	return error

def label_smoothing(y, e):
	'''
	Shaves off a small amount of probability from y, which is a dirac delta distribution, and redistributes it to the other classes.
	'''
	even = torch.full(y.shape, 1 / y.shape[0])
	return y * (1 - e) + e * even

def binarize(i: int, n: int):
	'''
	returns a list of length n, with all zeros except at position i
	'''
	if DEBUG:
		print('binarizing', i, 'of', n, '\n')
	output = torch.zeros((n,))
	output[ i ] = 1
	return output

def get_keys(table):
	'''
	returns all of the ids in the table as a sequence.
	'''
	table = pd.read_csv(table)
	table = table.set_index('Unnamed: 0')
	ids = [int(i) for i in table.index]
	return ids

def get_model(module, reset):
	'''
	loads the model source code and pretrained parameters if any.
	'''

	module = importlib.import_module(module)
	network = None
	if reset:
		network = module.Network()
		if DEBUG:
			print('network is', *[f'{i.shape}\n' for i in network.parameters()])
		network.inputdim = 391
		return network
	else:
		network = module.Network()
		path = Path(f'neural_network/Model.json')
		model_parameters = json.load(path.open(mode='r'))
		for key, value in model_parameters.items():
			if key == 'name':
				continue
			model_parameters[key] = torch.nn.parameter.Parameter(torch.tensor(value), requires_grad = True)
		network.load_state_dict(model_parameters)
		if DEBUG:
			print('network is', *[f'{i.shape}\n' for i in network.parameters()])
		network.inputdim = 391
		return network


if __name__ == '__main__':
	import sys
	source = sys.argv[1]
	table = sys.argv[2]
	model = sys.argv[3]
	batchsize = int(sys.argv[4])
	reset = bool(int(sys.argv[5])) # user is expected to specify this in the shell script as a zero or a one for false and true respectively
	epochs = int(sys.argv[6])

	keys = get_keys(table)
	dataset = Dataset.Dataset(source, keys)
	print("length of data:", len(dataset))
	sampler = Dataset.Sampler(dataset, batchsize)
	print("number  of batches: ", len(sampler))
	collate = Dataset.CollateFunction()

	dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=False, collate_fn=collate)
	network = get_model(model, reset)
	parameters = sum([torch.numel(p) for p in network.parameters()])
	print("total number of parameters:", parameters)

	train(dataloader, network, epochs, parameters)

