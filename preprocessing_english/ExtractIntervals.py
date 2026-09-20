'''
Provided a text file containint intervals formatted in a certain way, extracts them and returns them as a list of integers.
'''

def extract_intervals(source):
	'''
	Reads the source file line by line and extracts the interval information, returning the result as a list of integer intervals
	'''
	input = open(source, mode='r')
	intervals = []
	for line in input:
		interval = extract_interval(line)
		intervals.append(interval)
	return intervals


def extract_interval(line):
	'''
	reads each character in the line and pulls out the first two integers delimited by whitespace, then returns them as a tuple.
	'''
	line = line.strip()
	interval = []
	wordbuf = []
	line = list(line)
	line.append('\0')
	i = 0
	c = line[i]
	i+=1
	digits = ['0','1','2','3','4','5','6','7','8','9']
	while(c != '\0'):
		if c in digits:
			while(c != ' ' and c != '\0'):
				wordbuf.append(c)
				c = line[i]
				i+=1
			interval.append(int(''.join(wordbuf)))
			wordbuf = []
		else:
			c = line[i]
			i+=1
	return interval
