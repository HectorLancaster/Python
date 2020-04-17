import matplotlib.pyplot as plt
import numpy as np
# Google plt hist to explore optional parameters.
# By default, hist uses 10 evenly spaced bins and tries to optomise both
# bin width and bin location.

x = np.random.normal(size = 1000)
#plt.hist(x)

# Setting density to True normalises the y-axis, giving the proportions of
# observations in each bin rather than the total number.
# The bins parameter is set to have 20 bins, that start at 5 and end at -5.
plt.hist(x, density=True, bins=np.linspace(-5,5,21));

# Gamma distribution
# gamma(a,b,c) is the number of rows, columns and data points respectivley
x1 = np.random.gamma(2,3,100000)

# Plots a figure
plt.figure()

# Plots histogram
# .subplot(nrows, ncols, index) enables the below histograms to be plotted as
# part of the above figure. The index is numbered left to right, row by row. 
plt.subplot(221)
plt.hist(x1, bins=30)

# Plots normalised histogram
plt.subplot(222)
plt.hist(x1, bins=30, density=True);

# Plots cumulative histogram
plt.subplot(223)
plt.hist(x1, bins=30, cumulative=True);

# Plots normalised cumulative histogram of type "step"
plt.subplot(224)
plt.hist(x1, bins=30, cumulative=True, density=True, histtype = "step");