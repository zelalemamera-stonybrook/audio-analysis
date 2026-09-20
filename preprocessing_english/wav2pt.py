'''
Reads a wav file from the source directory, translates it to a tensor, then writes it to the target directory.
'''
import torch
import argparse
import torchaudio
import torchcodec
from pathlib import Path

def wav2pt(source, target):
	'''
	The source is an audio file in wav format, the target will be a tensor in pt format.
	'''
	waveform, samplerate = torchaudio.load(source)
	torch.save(waveform.reshape(-1), target)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("source")
	parser.add_argument("target")
	args = parser.parse_args()
	wav2pt(Path(args.source), Path(args.target))
