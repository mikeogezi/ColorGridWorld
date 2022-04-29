import numpy as np
import sys

_3_file = '3x3_enum'
_2_file = 'enum'

windows = [
	(0, 1),
	(0, 25),
	(25, 50),
	(50, 75),
	(75, 99),
	(99, 100),
]

if len(sys.argv) < 2:
	print('Invalid command line args')
	exit(1)

r = ['wins' in i for i in open(sys.argv[-1], 'r').readlines()]
print('{} episodes, {} wins'.format(len(r), np.count_nonzero(r)))
for l, u in windows:
	i = int((l/100.)*len(r))
	j = int((u/100.)*len(r))
	print('{}% - {}%: {:.2f} [{}:{}]'.format(l, u, np.count_nonzero(r[i:j])/len(r[i:j])*100, i, j))
