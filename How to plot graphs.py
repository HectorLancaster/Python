# First graph plotting code

# Import the library that is used to produce graphs
import matplotlib.pyplot as plt

# Use the .plot() function to plot the data to a graph,
# Without another data axis, this plots the x as the indexes (0,1,2,3,4)
plt.plot([0,1,4,9,16])

# The inclusion of a semi-colon at the end stops python printing certain text
plt.plot([0,1,4,9,16]);


# Import numpy to help create the variable x
import numpy as np

# This creates an array that starts at 0, ends at 10 and has 20 values
x = np.linspace(0, 10, 20)

# Create an array of the square of x
y = x**2

# The variables are shown below
x
#Out[8]: 
#array([ 0.        ,  0.52631579,  1.05263158,  1.57894737,  2.10526316,
#        2.63157895,  3.15789474,  3.68421053,  4.21052632,  4.73684211,
#        5.26315789,  5.78947368,  6.31578947,  6.84210526,  7.36842105,
#        7.89473684,  8.42105263,  8.94736842,  9.47368421, 10.        ])

y
#Out[9]: 
#array([  0.        ,   0.27700831,   1.10803324,   2.49307479,
#         4.43213296,   6.92520776,   9.97229917,  13.5734072 ,
#        17.72853186,  22.43767313,  27.70083102,  33.51800554,
#        39.88919668,  46.81440443,  54.29362881,  62.32686981,
#        70.91412742,  80.05540166,  89.75069252, 100.        ])

# To plot data with x and y coordinate, write as below
plt.plot(x,y)

# Create two new variables y1 and y2
y1 = x**2
y2 = x**1.5

# The "bo-" defines the plot's properties. "b" is the colour, blue
# "o" is the shape of the data points, circles
# and "-" is the type of line, straight.
plt.plot(x, y1, "bo-")

# The keywords linewidth and markersize are self evident
plt.plot(x, y1, "bo-", linewidth=2, markersize=4)
plt.plot(x, y1, "bo-", linewidth=2, markersize=12)

# here "g" means green and "s" square
plt.plot(x, y2, "gs-", linewidth=2, markersize=12)