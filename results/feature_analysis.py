'''
The following python program generates statistics on feature importance scores. The hypothesis anaysis file is used to colect the importance score for every feature across all the syllables
in the development dataset, and this is used to produce the maximum, minimum, and median importance score for that feature. One possible theory is that features which have high importance scores
across many syllables are considered more important for the learning task in general, whereas features which have very few high importance scores or low importance scores across the data are considered
less important for the learning task.
'''
import pandas as pd
from pathlib import Path
import argparse
import torch
from tab2csv import tab2csv
from Filefunctions import rename
import re

def analyze_importance(source: str, target: str):
	'''
	Reads in the table from source, extracts the feature column and generates relevenat statistics over this table.
	'''
	csvsource = rename(source, '.csv')
	tab2csv(source, csvsource)
	table = pd.read_csv(Path(csvsource))
	outputstream = target.open(mode = 'w')

	print(table.head(10))
	rows = []
	for i in table.index:
		string = ''
		for c in table['features'][i]:
			if(c == '[' or c == ']' or c == ' '):
				continue
			string += c
		tokens = re.split(',', string)
		tokens = [float(token) for token in tokens]
		rows.append(tokens)

	outputstream.write('feature\tmax\tmin\tmedian\tmean\n')
	for j in range(len(rows[0])):
		tensor = []
		for i in range(len(rows)):
			tensor.append(rows[i][j])
		tensor = torch.tensor(tensor)
		max, min, median, mean = round(torch.max(tensor).item(),3), round(torch.min(tensor).item(),3), round(torch.median(tensor).item(),3), round(torch.mean(tensor).item(),3)
		outputstream.write(f'{j}\t{max}\t{min}\t{median}\t{mean}\n')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("source")
	parser.add_argument("target")
	args = parser.parse_args()
	analyze_importance(Path(args.source), Path(args.target))
