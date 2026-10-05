'''
the following program splits the data table provided in the source directory into three batches of size 80 10 10 % of the whole respectively.
the batches are mutually exclusive and will be labelled train test and dev sets and saved to the target directory under folders with that name.
'''

import argparse
from pathlib import Path
import torch
from PrepareAlignments import split_data
import pandas as pd
import os
seed = 3529145006359120161



def splittables(source: str, target: str):
	'''
	takes the table located at source and splits it into the size specified above. then it saves each of the split tables to the directory target
	by the names specified above.
	'''
	table = pd.read_csv(source)
	data = {'data':table}
	split = split_data(data)
	for key, value in split.items():
		path = os.path.join(target, Path(key), Path('table.csv'))
		value.to_csv(path)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("source")
	parser.add_argument("target")
	args = parser.parse_args()
	splittables(Path(args.source), Path(args.target))
