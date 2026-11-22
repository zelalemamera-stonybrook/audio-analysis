'''
wav2vec 2016 enables researchers to obtain quality high dimensional embeddings of audio. The following program downloads the model from torchaudio and
transforms the input dataset to be respresented as embeddings from the final layer of the model.
'''
from pathlib import Path
import torch
import torchaudio
import torchcodec
import argparse
import os
import re
verbose = True

def get_embedding(source: str, target: str):
	'''
	for each sound vector in source, passes it through wav2vec and stores the final layer output into the target directory by the same name
	'''
	bundle = torchaudio.pipelines.WAV2VEC2_BASE
	model = bundle.get_model()
	model.eval()
	for file in source.iterdir():
		print("embedding", file, "...")
		waveform, samplerate = torchaudio.load(file)
		if waveform.shape[-1] < 500:
			waveform = torch.cat((waveform.reshape(-1), torch.zeros((500,))))
			waveform = waveform.reshape((1, waveform.shape[-1]))
		if verbose:
			print("extracted sound vector:", waveform.shape)
		with torch.no_grad():
			vecs, _ = model.extract_features(waveform)
			embedding = vecs[-1].reshape(-1)
			filename = os.path.split(file)[-1]
			filename = rename(filename, '.pt')
			torch.save(embedding, os.path.join(target, filename))

def rename(filename: str, extension: str):
	'''
	uses a regular pattern to rename filename's extension to the specified ending
	'''
	raw_name = re.split(r'\.', str(filename))[0]
	return Path(f'{raw_name}{extension}')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("source")
	parser.add_argument("target")
	args = parser.parse_args()
	get_embedding(Path(args.source), Path(args.target))
