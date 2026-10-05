'''
A repository of commonly used file functions so that they can be reused by other programs
'''

import re
from pathlib import Path

def rename(filename: str, extension: str):
	'''
	uses a regular pattern to rename filename's extension to the specified ending
	'''
	raw_name = re.split(r'\.', str(filename))[0]
	return Path(f'{raw_name}{extension}')
