'''
The following program summarizes the results in the statistics tables for presentation.
'''


import argparse
from pathlib import Path
import pandas as pd

def summarize(target: str, *tables):
	'''
	goes through each table and generates a summary of the statistics in the table. The result is written as one row in a table saved to target.
	Currently, only the row with the maximum fscore value is kept.
	'''
	if target.exists():
		targetstream = target.open(mode='a')
	else:
		targetstream = target.open(mode='w')
		targetstream.write('data\tepochs\tprecision\trecall\tfscore\n')
	for path in tables:
		table = pd.read_csv(path)
		name = table['data'][0]
		fscore = table['fscore']
		i = fscore.argmax()
		epoch, precision, recall, fscore = table['epochs'][i], table['precision'][i], table['recall'][i], table['fscore'][i]
		targetstream.write(f'{name}\t{epoch}\t{precision}\t{recall}\t{fscore}\n')




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("target")
	parser.add_argument("tables", nargs='*')
	args = parser.parse_args()
	summarize(Path(args.target), *[Path(t) for t in args.tables])
