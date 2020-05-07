
# Import libraries.
import matplotlib.pyplot as plt
import numpy as np


# Open mapping file and save each line in file_data.
with open("\\Users\\Hector\\Desktop\\Data\\KC10 map.txt", "r") as KC10:
              file_data = KC10.readlines()


# Split each line in file data where a space is found and assign each variable
# to a list, each line's list is then nested in the list 'spectrum'.
spectrum = [line.split() for line in file_data]
# Set variable 'length' for the length of the list 'spectrum'.
length = len(spectrum) 
# Set variables names for each column in 'spectrum'.
x = [int(spectrum[i][0]) for i in range(length)]
y = [int(spectrum[i][1]) for i in range(length)]
wavenumber = [float(spectrum[i][2]) for i in range(length)]
intensity = [float(spectrum[i][3]) for i in range(length)]

# Set variable data_points to the number of zeros in x, and thus the number
# of data points per xy coordinate.
data_points = x.count(0)
# Initialise spectra as an empty dictionary.
spectra = dict()
# Loop to cycle through each coordinate on map (a,b) corresponding to (x,y).
for a in range(0,105,5):            
    for b in range(-100,5,5):
        # Initialise data as a list.
        data = list()
        # Loop to append each wavenumber and intensity coordinate to the 
        # library key 'a, b' and so seperating data into individual spectra.
        for i in range(length):
            if x[i] == a and y[i] == b:
                data.append((wavenumber[i], intensity[i]))
        spectra[a,b] = data

# Find the shape of each spectral array and the length of the spectral 
# dictionary.
spectra_shape = list(np.array(spectra[0, -100]).shape)
length_dict = len(spectra)
# Initialise an array of zeros with spectra_shape.
sum_array = np.zeros(spectra_shape)
# Cycle through each dictionary key.
for a in range(0,105,5):
    for b in range(-100,5,5):
        # Sum all arrays in dictionary.
        sum_array = sum_array + np.array(spectra[a,b])
# Find avarage of all arrays.
avspectra = sum_array/(length_dict)

# Plot avspectra and label.
plt.plot([Column[0] for Column in avspectra],
         [Column[1] for Column in avspectra], label = "KC10")

# Label axes.         
plt.xlabel("Wavenumber")
plt.ylabel("Intensity")

# plt.axis([xmin,xmax,ymin,ymax]) sets the bounds of the axes.
plt.axis([1100, 1700, 150, 450])

# This is where the legend will be displayed.
plt.legend(loc="upper right")

# plt.savefig("average.pdf")

# 20.04.20 taking 18s to run script