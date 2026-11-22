'''
This function exists solely to shrink.
'''
import torch

def shrink(x, amount):
	'''
	It shrinks x by amount
	'''
	v = torch.load(x)
	v = v[:,:,:amount]
	torch.save(v, x)


if __name__ == '__main__':
	import sys
	x = sys.argv[1]
	amount = int(sys.argv[2])
	shrink(x, amount)
