import matplotlib.pyplot as plt
import numpy as np

#----------------------------Import Data--------------------------------------
with open("\\Users\\Hector\\Desktop\\Data\\KC10 map.txt", "r") as KC10:
              KC10_raw = KC10.readlines()
with open("\\Users\\Hector\\Desktop\\Data\\LiC10 map.txt", "r") as LiC10:
              LiC10_raw = LiC10.readlines()
with open("\\Users\\Hector\\Desktop\\Data\\yp50 map.txt", "r") as yp50:
              yp50_raw = yp50.readlines()

#-----------------------------Sort Data---------------------------------------
spectrum = [line.split() for line in KC10_raw]
length = len(spectrum) 
x_KC10 = [int(spectrum[i][0]) for i in range(length)]
y_KC10 = [int(spectrum[i][1]) for i in range(length)]
wavenumber_KC10 = [float(spectrum[i][2]) for i in range(length)]
intensity_KC10 = [float(spectrum[i][3]) for i in range(length)]

data_points = x_KC10.count(0)
spectra_KC10 = dict()
for a in range(0,105,5):            
    for b in range(-100,5,5):
        data = list()
        for i in range(length):
            if x_KC10[i] == a and y_KC10[i] == b:
                data.append((wavenumber_KC10[i], intensity_KC10[i]))
        spectra_KC10[a,b] = data


spectrum = [line.split() for line in LiC10_raw]
length = len(spectrum) 
x_LiC10 = [int(spectrum[i][0]) for i in range(length)]
y_LiC10 = [int(spectrum[i][1]) for i in range(length)]
wavenumber_LiC10 = [float(spectrum[i][2]) for i in range(length)]
intensity_LiC10 = [float(spectrum[i][3]) for i in range(length)]

data_points = x_LiC10.count(0)
spectra_LiC10 = dict()
for a in range(0,105,5):            
    for b in range(-100,5,5):
        data = list()
        for i in range(length):
            if x_LiC10[i] == a and y_LiC10[i] == b:
                data.append((wavenumber_LiC10[i], intensity_LiC10[i]))
        spectra_LiC10[a,b] = data

    
spectrum = [line.split() for line in yp50_raw]
length = len(spectrum) 
x_yp50 = [int(spectrum[i][0]) for i in range(length)]
y_yp50 = [int(spectrum[i][1]) for i in range(length)]
wavenumber_yp50 = [float(spectrum[i][2]) for i in range(length)]
intensity_yp50 = [float(spectrum[i][3]) for i in range(length)]
        
data_points = x_yp50.count(0)
spectra_yp50 = dict()
for a in range(0,105,5):            
    for b in range(-100,5,5):
        data = list()
        for i in range(length):
            if x_yp50[i] == a and y_yp50[i] == b:
                data.append((wavenumber_yp50[i], intensity_yp50[i]))
        spectra_yp50[a,b] = data
        
#------------------------------Averages---------------------------------------
spectra_shape = list(np.array(spectra_KC10[0, -100]).shape)
length_dict = len(spectra_KC10)
sum_array = np.zeros(spectra_shape)
for a in range(5,105,5):
    for b in range(-95,5,5):
        sum_array = sum_array + np.array(spectra_KC10[a,b])
KC10_avspectra = sum_array/(length_dict)

spectra_shape = list(np.array(spectra_LiC10[0, -100]).shape)
length_dict = len(spectra_LiC10)
sum_array = np.zeros(spectra_shape)
for a in range(5,105,5):
    for b in range(-95,5,5):
        sum_array = sum_array + np.array(spectra_LiC10[a,b])
LiC10_avspectra = sum_array/(length_dict)

spectra_shape = list(np.array(spectra_yp50[0, -100]).shape)
length_dict = len(spectra_yp50)
sum_array = np.zeros(spectra_shape)
for a in range(5,105,5):
    for b in range(-95,5,5):
        sum_array = sum_array + np.array(spectra_yp50[a,b])
yp50_avspectra = sum_array/(length_dict)

#------------------------------Plotting---------------------------------------

KC10_max = max([column[1] for column in KC10_avspectra])
norm_KC10 = [column[1]/KC10_max for column in KC10_avspectra]

LiC10_max = max([column[1] for column in LiC10_avspectra])
norm_LiC10 = [column[1]/LiC10_max for column in LiC10_avspectra]

yp50_max = max([column[1] for column in yp50_avspectra])
norm_yp50 = [column[1]/yp50_max for column in yp50_avspectra]


plt.plot([column[0] for column in yp50_avspectra],
         norm_yp50,
         "r-", linewidth=1, label = "yp50")

plt.plot([column[0] for column in LiC10_avspectra],
         norm_LiC10,
         "g-", linewidth=1, label = "LiC10")

plt.plot([column[0] for column in KC10_avspectra],
         norm_KC10,
         "b-", linewidth=1, label = "KC10")


plt.xlabel("Raman shift (cm⁻¹)")
plt.ylabel("Intensity (arb. units)")

plt.axis([1050, 1575, 0.3, 1.3])

plt.legend(loc="upper left")

# plt.savefig("composite average map.pdf")

# ~45 seconds to run