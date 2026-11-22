'''
The following program  splits the table in question.
'''
import argparse
import torch
import pandas as pd
from pathlib import Path
import os
import shutil
import re



def split(source: str, table: str, target: str):
	'''
	splits the source directory into target by using the table provided
	'''
	table = pd.read_csv(table)
	table = table.set_index('Unnamed: 0')
	subset = set(table.index)
	for file in source.iterdir():
		filename = os.path.split(file)[-1]
		if getid(filename) in subset:
			print(file)
			shutil.copy(file, os.path.join(target, filename))
			print(target, filename)

def getid(file: str):
	'''
	removes the extension and any subindices of the filename and returns the result as an integer
	'''
	return  int(re.split('_', re.split(r'\.', str(file))[0])[0])

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("source")
	parser.add_argument("table")
	parser.add_argument("target")
	args = parser.parse_args()
	split(Path(args.source), Path(args.table), Path(args.target))
