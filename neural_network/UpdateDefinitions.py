'''
This program writes definitions to the definitions file in this directory. All of the inputs that are given are updated.
'''

def write_definition(N, N_LINEAR, N_HEADS, OUTPUT_DIM, SEQ_LEN, F_LEN, BATCH_SIZE, MIN, MAX):
	stream = open('neural_network/definitions.py', mode = 'w')
	stream.write(f'N = {N}\n')
	stream.write(f'N_LINEAR = {N_LINEAR}\n')
	stream.write(f'N_HEADS = {N_HEADS}\n')
	stream.write(f'OUTPUT_DIM = {OUTPUT_DIM}\n')
	stream.write(f'SEQ_LEN = {SEQ_LEN}\n')
	stream.write(f'F_LEN = {F_LEN}\n')
	stream.write(f'BATCH_SIZE = {BATCH_SIZE}\n')
	stream.write(f'MIN = {MIN}\n')
	stream.write(f'MAX = {MAX}\n')

if __name__ == '__main__':
	import sys
	N = int(sys.argv[1])
	N_LINEAR = int(sys.argv[2])
	N_HEADS = int(sys.argv[3])
	OUTPUT_DIM = int(sys.argv[4])
	SEQ_LEN = int(sys.argv[5])
	F_LEN = int(sys.argv[6])
	BATCH_SIZE = int(sys.argv[7])
	MIN = float(sys.argv[8])
	MAX = float(sys.argv[9])

	write_definition(N, N_LINEAR, N_HEADS, OUTPUT_DIM, SEQ_LEN, F_LEN, BATCH_SIZE, MIN, MAX)

