'''
The following python program transforms a tab separated document into a comma separated document. This is a simple algorithm which replaces all of the tabs by commas.
'''
from pathlib import Path
import argparse

def tab2csv(source: str, target: str):
	'''
	writes all of the characters in source to target with the exception of tabs which are rewritten to commas.
	'''
	input_stream = source.open(mode = 'r')
	output_stream = target.open(mode = 'w')
	c = input_stream.read(1)
	while(c != ''):
		if( c == '\t'):
			output_stream.write(',')
			c = input_stream.read(1)
			continue
		if(c == '['):
			output_stream.write('\"')
			output_stream.write(c)
			c = input_stream.read(1)
			continue
		if(c == ']'):
			output_stream.write(c)
			output_stream.write('\"')
			c = input_stream.read(1)
			continue
		output_stream.write(c)
		c = input_stream.read(1)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("source")
	parser.add_argument("target")
	args = parser.parse_args()
	tab2csv(Path(args.source), Path(args.target))
