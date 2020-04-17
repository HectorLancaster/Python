

import numpy as np

x = np.array([1,2,3])

y = np.array([2,4,6])

X = np.array([[1,2,3],[4,5,6]])

Y = np.array([[2,4,6],[8,10,12]])

x[2]
Out[5]: 3

x[0:2]
Out[6]: array([1, 2])

x + y
Out[7]: array([3, 6, 9])

z = x + y

X[:,1]
Out[9]: array([2, 5])

Y[:,1]
Out[10]: array([ 4, 10])

Y[:,1] + X[:,1]
Out[11]: array([ 6, 15])

X[1,:]
Out[12]: array([4, 5, 6])

X[1,:] + Y[1,:]
Out[13]: array([12, 15, 18])

X[1]
Out[14]: array([4, 5, 6])

[2,4] + [4,8]
Out[15]: [2, 4, 4, 8]

np.array([2,4]) + np.array([4,8])
Out[16]: array([ 6, 12])