'''
the following program finds the maximum length vector in the input directory and pads all of the vectors to that size. The result is saved to the target
directory provided.
'''
import argparse
import torch
from torch import Tensor
import torchaudio
import torchcodec
from pathlib import Path
import os
import re
verbose = False
mps = torch.device('mps')

def pad(source: str, target: str):
	'''
	identifies the maximum length vector in the source directory, and pads every vector to this number.
	then saves the result to the target directory
	'''
	max = 0
	for file in source.iterdir():
		vector = torch.load(file)
		print("looking for maximum", file, "size", len(vector))
		if len(vector) > max:
			max = len(vector)
	for file in source.iterdir():
		print("padding", file, "...")
		vector = torch.load(file)
		vector = zeropad(vector, max)
		print('new shape of tensor', vector.shape)
		filename = os.path.split(file)[-1]
		torch.save(vector, os.path.join(target, filename))

def rename(filename: str, extension: str):
        '''
        uses a regular pattern to rename filename's extension to the specified ending
        '''
        raw_name = re.split(r'\.', str(filename))[0]
        return Path(f'{raw_name}{extension}')

def zeropad(waveform: Tensor, max: int):
	'''
	adds a max number of zeros from the left and the right
	'''
	waveform = waveform.reshape(-1)
	if verbose:
		print('received', waveform.shape)
		print('padding to', max)
	padsize = max - len(waveform)
	pad1 = torch.zeros((padsize//2, ), device=mps)
	pad2 = torch.zeros((padsize//2, ), device=mps)
	waveform = torch.cat((pad1, waveform, pad2))
	if padsize % 2 == 1:
		waveform = torch.cat((waveform, torch.tensor([0], device=mps)))
	#print('padded vector', waveform.shape)
	return waveform



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("source")
	parser.add_argument("target")
	args = parser.parse_args()
	source = args.source
	target = args.target
	pad(Path(source), Path(target))
