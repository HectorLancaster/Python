

import matplotlib.pyplot as plt
import numpy as np

datasets = ["KC10", "LiC10", "yp50"]
num_datasets = len(datasets)
data = list()

for i in range(num_datasets):
    with open("\\Users\\Hector\\Desktop\\Data\\" + datasets[i] + \
              " map.txt", "r") as datasets[i]:
              data.append(datasets[i].readlines())

spectrum = [line.split() for line in data[i]]
length = len(spectrum) 
x = [int(spectrum[i][0]) for i in range(length)]
y = [int(spectrum[i][1]) for i in range(length)]
wavenumber = [float(spectrum[i][2]) for i in range(length)]
intensity = [float(spectrum[i][3]) for i in range(length)]

data_points = x.count(0)
spectra = dict()
for a in range(0,105,5):            
    for b in range(-100,5,5):
        data = list()
        for i in range(length):
            if x[i] == a and y[i] == b:
                data.append((wavenumber[i], intensity[i]))
        spectra[a,b] = data

spectra_shape = list(np.array(spectra[0, -100]).shape)
length_dict = len(spectra)
sum_array = np.zeros(spectra_shape)
for a in range(5,105,5):
    for b in range(-95,5,5):
        sum_array = sum_array + np.array(spectra[a,b])
avspectra = sum_array/(length_dict)

plt.plot([Column[0] for Column in avspectra],
         [Column[1] for Column in avspectra], label = "KC10")

plt.xlabel("Wavenumber")
plt.ylabel("Intensity")

plt.axis([1050, 1575, 150, 450])

plt.legend(loc="upper right")

# plt.savefig("average.pdf")
