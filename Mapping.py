

with open("\\Users\\Hector\\Desktop\\Data\\KC10 map.txt", "r") as KC10:
              file_data = KC10.readlines()

import matplotlib.pyplot as plt
import numpy as np

spectrum = [line.split() for line in file_data]
length = len(spectrum)
x = [int(spectrum[i][0]) for i in range(length)]
y = [int(spectrum[i][1]) for i in range(length)]
wavenumber = [float(spectrum[i][2]) for i in range(length)]
intensity = [float(spectrum[i][3]) for i in range(length)]

data_points = x.count(0)
spectra_dictionary = dict()
for a in range(0,105,5):
    for b in range(-100,5,5):
        data = list()
        for i in range(length):
            if x[i] == a and y[i] == b:
                data.append((wavenumber[i], intensity[i]))
        spectra_dictionary[a,b] = data


sum1 = np.zeros([576,2])
for a in range(5,105,5):
    for b in range(-95,5,5):
        sum1 = sum1 + np.array(spectra_dictionary[a,b])
average = sum1/441


plt.plot([Column[0] for Column in average],
         [Column[1] for Column in average])

plt.savefig("average.pdf")
