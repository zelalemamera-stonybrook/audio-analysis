'''
The following program translates a specially formatted text file into a csv and saves it to the provided target by a csv extension.
'''

import argparse
from pathlib import Path

def txt2csv(source: str, target: str):
	'''
	streams source into target while replacing any tab characters with commas.
	'''
	sourcestream = source.open(mode='r')
	targetstream = target.open(mode='w')
	c = sourcestream.read(1)
	while(c != ''):
		if c == '\t':
			targetstream.write(',')
			c = sourcestream.read(1)
			continue
		if c == '(':
			targetstream.write('\"')
			targetstream.write(c)
			c = sourcestream.read(1)
			continue
		if c == ')':
			targetstream.write(c)
			targetstream.write('\"')
			c = sourcestream.read(1)
			continue
		targetstream.write(c)
		c = sourcestream.read(1)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("source")
	parser.add_argument("target")
	args = parser.parse_args()
	txt2csv(Path(args.source), Path(args.target))
