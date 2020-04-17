# semilogx() plots x on a log scale and y on it's original scale
# semilogy() plots y on a log scale and x on it's original scale
# loglog() plots both x and y on the log scale
# The default base is 10 for logs

# Import libraries
import matplotlib.pyplot as plt
import numpy as np

# Create datasets (1D array) of logarithmic spacing
x = np.logspace(-1, 1, 40)
y1 = x**2.0
y2 = x**1.5

# Plot x and y data on a log scale
plt.loglog(x, y1, "bo-", linewidth=2, markersize=12, label="First")
plt.loglog(x, y2, "gs-", linewidth=2, markersize=12, label="Second")

# LaTeX is used to alter phonts, if "$X$" then the X is in italics
# This labels the y and x axes
plt.xlabel("$X$")
plt.ylabel("Y")

# plt.axis([xmin,xmax,ymin,ymax]) sets the bounds of the axis
# plt.axis([-0.5, 10.5, -5, 105])

# This is where teh legend will be displayed
plt.legend(loc="upper left")

# This is the name of the file, save location and file type,
# the directory can be input here, remember \\ escape sequence
plt.savefig("myplotlog2.pdf")