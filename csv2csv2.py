'''
Small function that rewrites the data column of table for better summary formatting.
'''


import argparse
from pathlib import Path

def translate(source: str, target: str):
	'''
	rewrites the data column entry in a specific way that is predetermined.
	'''
	inputstream = source.open(mode='r')
	targetstream = target.open(mode='w')
	c = inputstream.read(1)
	while(c != ''):
		if c == 'W':
			while(c != 'c'):
				targetstream.write(c)
				c = inputstream.read(1)
			targetstream.write(c)
			targetstream.write('+PraatModel')
			c = inputstream.read(1)
			continue
		targetstream.write(c)
		c = inputstream.read(1)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("source")
	parser.add_argument("target")
	args = parser.parse_args()
	translate(Path(args.source), Path(args.target))
