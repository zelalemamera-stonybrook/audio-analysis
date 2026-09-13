'''
The following program tests the performance of Model. It generates f score, accuracy, precision, and recall over the provided dataset.
in addition, this program also presents the weights that the neural network learned for feature embeddings. This can be used to make a judgement about which features are important.
'''
import json
from pathlib import Path
import torch
import torchaudio
from torch import Tensor
import pandas as pd
import argparse
import re
import os
from pandas import DataFrame
from TrainNetwork import binarize
import importlib
from TrainNetwork import getmaxfeaturedimension
from Pad import zeropad
from Train import swap_dimensions, feature_encoder
SEED = 3529145006359120161
DEBUG = False

def test(network, hypothesis: str, table: str, results: str, epochs: int, *features):
	'''
	tests the network's performance on the feature data. The transformed data is saved to hypothesis, then additional statistics are generated from this to be written
	to results.
	'''
	if DEBUG:
		print('table is', table)
	table = pd.read_csv(table)
	table = table.set_index('Unnamed: 0')
	maxfeaturedimension = getmaxfeaturedimension(features)
	for i in table.index:
		address, wordembeddings = getfeature(i, features, maxfeaturedimension)
		if DEBUG:
			print('input for attention network is',wordembeddings.shape)
		y_hat = network.forward(wordembeddings)
		if DEBUG:
			print('forward complete')
			print('y_hat', y_hat)
		writeoutput(address, y_hat, hypothesis)
	accuracy, precision, recall, fscore = generatestatistics(hypothesis, table)
	write_statistics(accuracy, precision, recall, fscore, epochs, results)
	write_hypothesis_analysis(hypothesis, table, results, network.attention_weights)

def getword(i: int, source: str):
	'''
	returns the ith word from the source. It also returns the addresses of the syllables of the word
	'''
	word = sorted(list(source.glob(f'{i}_*')))
	return [torch.load(i) for i in word], word

def getfeature(i: int, features: tuple, max: int):
	'''
	gets the ith word in each of the directories in features
	'''
	global SEED
	g_cpu = torch.Generator()
	output = []
	word = None
	for j, featuredir in enumerate(features):
		word = sorted(list(featuredir.glob(f'{i}_*')))
		tensors = [torch.load(f) for f in word]
		if type(tensors[0]) == float:
			tensors = [torch.tensor([k]) for k in tensors]
		output.append([feature_encoder(j + 1, max) + zeropad(f, max) for f in tensors])
	return word, swap_dimensions(output)

def writeoutput(address: list, y_hat: list, hypothesis: str):
	'''
	writes the models prediction to the hypothesis directory by using the filnames provided in address
	'''
	for i, path in enumerate(address):
		filename = os.path.split(path)[-1]
		value = y_hat[i]
		if DEBUG:
			print('writing to ', hypothesis, filename)
		torch.save(value, os.path.join(hypothesis, filename))

def generatestatistics(source: str, table: DataFrame):
	'''
	source is a directory containing the output of the model's prediction on the testing set. The ouputs are for syllables only. The table is used to generate global statistics on both syllable and word
	information in the source.
	'''
	syllabletarget, wordtarget = get_gold_distribution(table)
	accuracy, precision, recall = getaccuracy(source, syllabletarget, wordtarget), getprecision(source, syllabletarget), getrecall(source, syllabletarget)
	fscore = getfscore(recall, precision)
	return accuracy, precision, recall, fscore

def get_gold_distribution(table: DataFrame):
	'''
	for each word in the table, it assigns it the sequence of correct distributions over the syllables and over the whole word.
	'''
	if DEBUG:
		print('building gold distribution')
		print(table.head(10))
	wordtarget = [(i, binarize(int(table['stress'][i] - 1), int(table['syllables'][i]))) for i in table.index]
	if DEBUG:
		print('generated word target')
		print('[', *wordtarget[:5], '...', *wordtarget[-5:], ']')
	syllabletarget = []
	for i, distribution in wordtarget:
		sylldistribution = []
		for j in distribution:
			sylldistribution.append(binarize(int(j), 2))
		syllabletarget.append( (i, sylldistribution))
	if DEBUG:
		print('generated syllable target')
		print('[', *syllabletarget[:5], '...', *syllabletarget[-5:], ']')
	return syllabletarget, wordtarget

def load_features(f: list):
	'''
	takes the features in f and concatenates them by syllable, so we can use the feature imbedding of the whole word.
	'''
	output = []
	syll = len(f[-1])
	for i in range(syll):
		vec = []
		for feature in f:
			vec += feature[i]
		output.append(torch.tensor(vec))
	return torch.stack(output)

def load_wav_to_vec(model, xpath: str):
	'''
	loads the wavtovec imbedding of x and returns it
	'''
	waveform = torch.tensor(json.load(xpath.open(mode='r')))
	vecs = []
	model.eval()
	with torch.no_grad():
		vecs, _ = model.extract_features(waveform)
	x = []
	for vec in vecs[-1]:
		x.append(vec.reshape(-1))
	x = torch.stack(x).detach()
	return x

def write_statistics(accuracy: tuple, precision: float, recall: float, fscore: float, epochs: int, results: str):
	'''
	writes these statistics to the folder results under the model's name
	'''
	print(f'writing result for {results}')
	path = Path(os.path.join(results, Path('statistics.txt')))
	if path.exists():
		with path.open(mode='a') as f:
			line = f'Praat\t{epochs}\t{accuracy}\t{precision}\t{recall}\t{fscore}\n'
			f.write(line)
	else:
		path.touch()
		with path.open(mode='w') as f:
			line = f'data\tepochs\taccuracy\tprecision\trecall\tfscore\n'
			f.write(line)
			line = f'Praat\t{epochs}\t{accuracy}\t{precision}\t{recall}\t{fscore}\n'
			f.write(line)

def write_hypothesis_analysis(hypothesis: str, table: DataFrame, results: str, attention_weights: list):
	'''
	the model's hypothesis is analysed by looking at the probabilites it generated for the source data and comparing them with the truth in the table.
	'''
	print(f'writing hypothesis analysis to {hypothesis}')
	if attention_weights == None or attention_weights == []:
		path = Path(os.path.join(results, Path('hypothesis_analysis.txt')))
		with path.open('w') as f:
			line = f'ipa\tsyllable\tprobability\tpredicted\tgold\tcorrect\tfeatures\n'
			f.write(line)
			for count, i in enumerate(table.index):
				word = sorted(list(hypothesis.glob(f'{i}_*')))
				ipa = table['ipa'][i]
				gold = binarize(int(table['stress'][i]) - 1, int(table['syllables'][i]))
				for j in range(len(word)):
					output = torch.load(word[j])
					if j == 0:
						line = f'{ipa}\t{j + 1}\t{to_list(output)}\t{torch.argmax(output)}\t{gold[j]}\t{ gold[j] == torch.argmax(output)}\tnone\n'
						f.write(line)
					else:
						line = f'\t{j + 1}\t{to_list(output)}\t{torch.argmax(output)}\t{gold[j]}\t{ gold[j] == torch.argmax(output)}\tnone\n'
						f.write(line)
		return
	path = Path(os.path.join(results, Path('hypothesis_analysis.txt')))
	if DEBUG:
		print('recieved attention weights', len(attention_weights))
	with path.open('w') as f:
		line = f'ipa\tsyllable\tprobability\tpredicted\tgold\tcorrect\tfeatures\n'
		f.write(line)
		for count, i in enumerate(table.index):
			word = sorted(list(hypothesis.glob(f'{i}_*')))
			ipa = table['ipa'][i]
			gold = binarize(int(table['stress'][i]) - 1, int(table['syllables'][i]))
			for j in range(len(word)):
				output = torch.load(word[j])
				if j == 0:
					line = f'{ipa}\t{j + 1}\t{to_list(output)}\t{torch.argmax(output)}\t{gold[j]}\t{ gold[j] == torch.argmax(output)}\t{[round(n, 3) for n in attention_weights[count][j]]}\n'
					f.write(line)
				else:
					line = f'\t{j + 1}\t{to_list(output)}\t{torch.argmax(output)}\t{gold[j]}\t{ gold[j] == torch.argmax(output)}\t{[round(n, 3) for n in attention_weights[count][j]]}\n'
					f.write(line)
def to_list(t: Tensor):
	'''
	converts this tensor to a list in a specific format that is desired for the output text file.
	'''

	rounded = [round(v.item(), 2) for v in t]
	return rounded

def getaccuracy(source: str, syllabletarget: list, wordtarget: list):
	'''
	generates accuracy of the source compared with both the syllable target and the wordtarget
	'''
	if DEBUG:
		print('generating accuracy')
	numerator = 0
	denominator = countsyllables(syllabletarget)
	for i, distribution in syllabletarget:
		word = sorted(list(source.glob(f'{i}_*')))
		if DEBUG:
			print('word is', word)
		word =  [torch.load(i) for i in word]
		for j, syll in enumerate(word):
			claim = binarize(torch.argmax(syll).item(), 2)
			true = distribution[j]
			if DEBUG:
				print('models claim is', claim, 'truth', true)
			if claim == true:
				if DEBUG:
					print('claim is true')
				numerator +=1
	syllaccuracy = numerator / denominator

	numerator = 0
	denominator = len(wordtarget)
	for i, distribution in wordtarget:
		word = sorted(list(source.glob(f'{i}_*')))
		if DEBUG:
			print('word is', word)
		word = [torch.load(i) for i in word]
		claim = [torch.argmax(i).item() for i in word]
		true = distribution
		if DEBUG:
			print('claim is', claim, 'truth', true)
		if claim == true:
			if DEBUG:
				print('claim is true')
			numerator +=1
	wordaccuracy = numerator / denominator

	return round(syllaccuracy, 3), round(wordaccuracy, 3)

def countsyllables(syllabletarget: list):
	'''
	a simple function that counts how many syllables are contained in total in the list
	'''
	counts = 0
	for i, lst in syllabletarget:
		counts += len(lst)
	return counts

def getprecision(source: str, syllabletarget: list):
	'''
	precision is computed over syllables and words for the dataset in source
	'''
	print('computing precision')
	numerator = 0
	denominator = 0
	for i, distribution in syllabletarget:
		word = sorted(list(source.glob(f'{i}_*')))
		if DEBUG:
			print('word is', word)
		word =  [torch.load(i) for i in word]
		for j, syll in enumerate(word):
			claim = binarize(torch.argmax(syll).item(), 2)
			true = distribution[j]
			if DEBUG:
				print('claim is', claim, 'truth', true)
			if claim == [0,1]:
				if DEBUG:
					print('claim is counted')
				denominator +=1
				if claim == true:
					if DEBUG:
						print('claim is correct')
					numerator +=1
	if denominator == 0:
		return 0
	syllableprecision = numerator / denominator
	return round(syllableprecision, 3)

def getrecall(source: str, syllabletarget: list):
	'''
	recall is computed over syllables and words for the dataset in source
	'''
	print('computing recall')
	numerator = 0
	denominator = 0
	for i, distribution in syllabletarget:
		word = sorted(list(source.glob(f'{i}_*')))
		if DEBUG:
			print('word is', word)
		word =  [torch.load(i) for i in word]
		for j, syll in enumerate(word):
			claim = binarize(torch.argmax(syll).item(), 2)
			true = distribution[j]
			if DEBUG:
				print('claim is', claim, 'truth', true)
			if true == [0,1]:
				if DEBUG:
					print('truth is counted')
				denominator +=1
				if claim == true:
					if DEBUG:
						print('claim is correct')
					numerator +=1
	syllablerecall = numerator / denominator
	return round(syllablerecall, 3)

def getfscore(recall: float, precision: float):
	'''
	computes the recall of the source
	'''
	if recall + precision == 0:
		return None
	f = round(2 * (recall * precision)/ (recall + precision), 2)
	return round(f, 3)

def recall(output: list[Tensor],  gold: list):
	'''
	computes the recall of output against gold.
	'''
	y_hat = []
	for word in output:
		classification_value = []
		for distribution in word:
			classification_value.append(torch.argmax(distribution))
		y_hat.append(classification_value)
	recall = 0
	for prediction, true in zip(y_hat, gold):
		i = torch.argmax(torch.tensor(true))
		if prediction[i] == 1:
			recall +=1
	return round(recall / len(gold), 2)

def precision(output: list[Tensor],  gold: list):
	'''
	computes the precision as the proportion of true positives to all positives
	'''
	y_hat = []
	for word in output:
		classification_value = []
		for distribution in word:
			classification_value.append(torch.argmax(distribution))
			y_hat.append(classification_value)
	precision = 0
	total = 0
	for prediction, true in zip(y_hat, gold):
		for i, val in enumerate(prediction):
			if val == 1:
				total +=1
				if true[i] == 1:
					precision+=1
	if total == 0:
		return 0
	return round(precision/total, 2)


def fscore(recall: float, precision: float):
	'''
	computes the fscore measure
	'''
	if recall + precision == 0:
		return None
	f = round(2 * (recall * precision)/ (recall + precision), 2)
	return f

def filter(features: dict, syll: int, i: int, feature_list:list[str]):
	'''
	returns the ith element of each list in features
	'''
	f = []
	for feature in feature_list:
		word = []
		for j in range(syll):
			word.append(features[f'data_{syll}'][f'{i}'][f'{j}'][f'{feature}'])
		f.append(word)
	return f

def split_by_syllable(input: list, gold: list, features: list, size_list: list[int]):
	'''
	splits the input batch by syllables, each batch is always sorted by syllable size, size list contains the number of words in each batch. The result is returned as a triple.
	'''
	two = size_list[0]
	three = size_list[1]
	four = size_list[-1]

	input = sorted(input, key = lambda x: (x[0], x[1]))
	gold = sorted(gold, key = lambda x: (x[0], x[1]))

	two_input, two_gold, two_features = input[:two], gold[:two], [features[f]['syllable_2'] for f in features.keys()]
	three_input, three_gold, three_features = input[two:two + three], gold[two:two + three], [features[f]['syllable_3'] for f in features.keys()]
	four_input, four_gold, four_features = input[two + three:], gold[two + three:], [features[f]['syllable_4'] for f in features.keys()]

	return (two_input, two_gold, two_features), (three_input, three_gold, three_features), (four_input, four_gold, four_features)

def count_syllable_sizes(input: list):
	'''
	the input contains a list of words that are already sorted by syllable size.
	we count how many of them are there in each and return the result as a list
	'''
	size_list = []
	path = Path(f'{input}/meta.json')
	meta = json.load(path.open(mode='r'))
	two_size = meta['syllable_2']
	three_size = meta['syllable_3']
	four_size = meta['syllable_4']
	size_list.append(two_size)
	size_list.append(three_size)
	size_list.append(four_size)
	return size_list


def load_model(model: str):
	'''
	loads in the trained model parameters and initializes it for testing.
	'''

	print('loading in model')
	path = Path(f'neural_network/{model}.json')
	if DEBUG:
		print('loading in', path)
	state_dict = json.load(path.open(mode='r'))
	for key, value in state_dict.items():
		if key == 'name':
			continue
		state_dict[key] = torch.nn.parameter.Parameter(torch.tensor(value), requires_grad = True)
	module = importlib.import_module(model)
	network = module.Network()
	network.load_state_dict(state_dict)
	network.eval()
	if DEBUG:
		print('network is\n', *[i.shape for i in network.parameters()])
	return network

if __name__== '__main__':
	'''
	model is the name of the module that has the source code of the model;
	table is the csv file containing the true labels of the source;
	hypothesis is a directory where the model's hypothesis will be stored;
	results is the directory where the model's performance on the source will be written;
	features is the list of feature directories that are used if the model uses attention.
	'''
	parser = argparse.ArgumentParser()
	parser.add_argument('model')
	parser.add_argument('hypothesis')
	parser.add_argument('table')
	parser.add_argument('results')
	parser.add_argument('epochs')
	parser.add_argument('features', nargs='*', default=None)
	args = parser.parse_args()
	network = load_model(args.model)

	test(network, Path(args.hypothesis), Path(args.table), Path(args.results), args.epochs, *[Path(i) for i in args.features])

