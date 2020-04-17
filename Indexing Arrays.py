import numpy as np

z1 = np.array([1,3,5,7,9])

z2 = z1 + 1

ind = [0,2,3]

z1[ind]
Out[5]: array([1, 5, 7])

ind = np.array([0,2,3])

z1[ind]
Out[7]: array([1, 5, 7])

z1 > 6
Out[8]: array([False, False, False,  True,  True])

z1[z1 > 6]
Out[9]: array([7, 9])

z2[z1 > 6]
Out[10]: array([ 8, 10])

ind = z1 > 6

ind
Out[12]: array([False, False, False,  True,  True])

z1[ind]
Out[13]: array([7, 9])

z2[ind]
Out[14]: array([ 8, 10])

w = z1[0:3]

w[0] = 3

w
Out[17]: array([3, 3, 5])

z1
Out[18]: array([3, 3, 5, 7, 9])

z1 = np.array([1,3,5,7,9])

ind = np.array([0,1,2])

w = z1[ind]

w
Out[22]: array([1, 3, 5])

w[0] = 3

w
Out[24]: array([3, 3, 5])

z1
Out[25]: array([1, 3, 5, 7, 9])