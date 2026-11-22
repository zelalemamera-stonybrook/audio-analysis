'''
This program picks a random subset of the specified size, then writes this subset to standard output.
'''
import pandas as pd

def extract(tableaddr, amount, target, seed):
	'''
	amount is assumed to be a real number between 0 and 1 representing the proportion of the table that needs to be sampled randomly. A seed is used in order to make the randomness repeatable.
	'''

	table = pd.read_csv(tableaddr)
	if "Unnamed: 0" in table.columns:
		table = table.set_index("Unnamed: 0")
	subset = table.sample(frac = amount, random_state = seed // 2 ** 30)
	for i in subset.index:
		print(i)
	subset.to_csv(target)
	remainder = table.drop(list(subset.index))
	remainder.to_csv(tableaddr)

if __name__ == '__main__':
	import sys
	table = sys.argv[1]
	amount = float(sys.argv[2])
	target = sys.argv[3]
	seed = int(sys.argv[4])
	extract(table, amount, target, seed)
