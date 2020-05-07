#--------------------------------import modules-------------------------------
import numpy as np
import matplotlib.pyplot as plt
import time

#---------------------------------start timer---------------------------------
start_time = time.process_time()

#--------------------------------define z-score-------------------------------
def z_score(intensity):
    mean_int = np.mean(intensity)
    std_int = np.std(intensity)
    z_scores = (intensity - mean_int) / std_int
    return z_scores

#-----------------------------------------------------------------------------
intensity = np.array([column[1] for column in spectra_KC10[5,-50]])
wavelength = np.array([column[0] for column in spectra_KC10[5,-50]])


dist = 0
delta_intensity = []
for i in np.arange(len(intensity) - 1):
    dist = intensity[i + 1] - intensity[i]
    delta_intensity.append(dist)


#-----------------------------------------------------------------------------
threshold = 3.5

#for a in range(0,105,5):
    #for b in range(-100,5,5):
            #intensity = np.array([column[1] for column in spectra_KC10[a,b]])
            #wavelength = np.array([column[0] for column in spectra_KC10[a,b]])
            #plt.plot(wavelength, intensity)




intensity_z_score = np.array(abs(z_score(intensity)))
plt.plot(wavelength, intensity_z_score)
plt.plot(wavelength, threshold*np.ones(len(wavelength)), label = "threshold")
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.xlabel("Wavelength" ,fontsize = 20)
plt.ylabel("Z-Score" ,fontsize = 20)
plt.show()


#---------------------------------plot spikes---------------------------------
# 1 is assigned to spikes, 0 to non-spikes:
spikes = abs(np.array(z_score(intensity))) > threshold

plt.plot(wavelength, spikes, color = "red")
plt.title("Spikes: " + str(np.sum(spikes)), fontsize = 20)
plt.grid()
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.xlabel("Wavelength" ,fontsize = 20)
plt.ylabel("Z-scores >" + str(threshold) ,fontsize = 20)
plt.show()
#---------------------------------end timer-----------------------------------
end_time = time.process_time()
print("Script runtime:", str(end_time - start_time), "\bs")
# last runtime = 0.15s