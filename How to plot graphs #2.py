# Add legend using legend()
# Adjust axis using axis()
# Set axis labels using xlabel(), ylabel()
# Save figure using savefig()

# Import libraries
import matplotlib.pyplot as plt
import numpy as np

# Create datasets (1D array)
x = np.linspace(0, 10, 20)
y1 = x**2.0
y2 = x**1.5

# Plot data, label="First" gives a label to the dataset in the plot
plt.plot(x, y1, "bo-", linewidth=2, markersize=12, label="First")
plt.plot(x, y2, "gs-", linewidth=2, markersize=12, label="Second")

# LaTeX is used to alter phonts, if "$X$" then the X is in italics
# This labels the y and x axes
plt.xlabel("$X$")
plt.ylabel("Y")

# plt.axis([xmin,xmax,ymin,ymax]) sets the bounds of the axis
plt.axis([-0.5, 10.5, -5, 105])

# This is where teh legend will be displayed
plt.legend(loc="upper left")

# This is the name of the file, save location and file type,
# the directory can be input here, remember \\ escape sequence
plt.savefig("myplot.pdf")