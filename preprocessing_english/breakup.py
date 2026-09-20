'''
takes a tensor and break it up into the intervals provided. Then saves these intervals into the target folder provided.
'''
import torch
import argparse
from pathlib import Path
import os
from ExtractIntervals import extract_intervals

def breakup(source, intervals, target):
	'''
	breaks up the source tensor in the intervals provided and saves the result to target
	'''
	intervals = extract_intervals(intervals)
	file = torch.load(source)
	breakup = []
	for x, y in intervals:
		breakup.append(file[x:y])
	for j, piece in enumerate(breakup):
		torch.save(piece, f'{target}/{j}.pt')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("source")
	parser.add_argument("intervals")
	parser.add_argument("target")
	args = parser.parse_args()
	breakup(Path(args.source), Path(args.intervals), Path(args.target))
