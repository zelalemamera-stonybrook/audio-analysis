'''
Defines a dataset object so that training programs can conveniently load, update, and check the status of data.
'''
import torch
DEBUG = False
device = torch.device('cpu')
torch.set_default_device(device)
torch.set_default_dtype(torch.float64)

class Dataset(torch.utils.data.Dataset):

	def __init__(self, source, keys):
		super().__init__()
		self.source = source
		self.keys = keys
		self.len = len(keys)

	def __len__(self):
		return self.len

	def __getitem__(self, id):

		x = torch.load(f'{self.source}/{id}/x.pt', map_location=device).to(torch.float64)
		y = torch.load(f'{self.source}/{id}/y.pt', map_location=device).to(torch.float64)
		mask = len(y)
		return x, y, mask


class CollateFunction():

	def __init__(self):
		pass

	def __call__(self, input_list):

		x_tensors = []
		y_tensors = []
		masks = []
		for x, y, m in input_list:
			x_tensors.append(x)
			y_tensors.append(y)
			masks.append(m)
		return torch.stack(x_tensors), y_tensors, masks


class Sampler(torch.utils.data.Sampler):

	def __init__(self, dataset, batchsize):
		super().__init__()
		self.dataset = dataset
		self.batchsize = batchsize
		self.len = len(self.dataset) // batchsize

	def __iter__(self):

		n = len(self.dataset) // self.batchsize
		for i in range(n):
			yield self.dataset.keys[i*self.batchsize: i*self.batchsize + self.batchsize]

	def __len__(self):
		return self.len

