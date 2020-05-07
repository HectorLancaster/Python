# Method adapted from: 
# Whitaker, D and Hayes, K. “A simple algorithm for despiking Raman spectra.”
# Chemometrics and Intelligent Laboratory Systems 179 (2018): 82–84.

#--------------------------------import modules-------------------------------
import numpy as np
import matplotlib.pyplot as plt
import time

#---------------------------------start timer---------------------------------
start_time = time.process_time()

#--------------------------------define z-score-------------------------------


def modified_z_score(y): # Find the modified z-score of a dataset y
    median = np.median(y)
    mad = np.median([np.abs(y-median)]) # Median absolute deviation (MAD)
    z = 0.6745 * (y - median) / mad # The z-score, 0.6745 adjusting factor
    return z

#--------------------------------define fixer---------------------------------


def fixer(y, z_bool, ma=5):
    n = len(z) # for indexing the arrays, (575,)
    y_out = y.copy()
    z_bool[0] = z_bool[n-1] = True # set the first and last value to True
    spikes = np.array(np.where(z_bool == True)) # gives spike locations
    spikes = np.reshape(spikes,(len(spikes[0]),)) # (1)
    for i in spikes: # uses the locations of spikes as i values
        w = np.arange(max(0, i - ma), min(n, i + 1 + ma)) # (2)
        w = w[z_bool[w] == 0] # returns the non-peak locations from the range
        y_out[i] = np.mean(y[w]) # (3)
    return y_out # (576,)

# (1) must reshape from (1,len) to (len,) for the cycling through i in the for
# loop to work.
# (2) sets a range around i, if close to array boundaries, limits range to
# boundary, +1 becuase the way python references does not include top value.
# (3) replace peak with mean of surrounding non-peak data.
#-----------------------------------------------------------------------------

x = np.array([column[0] for column in spectra_KC10[0,-100]]) # for plotting
y = np.array([column[1] for column in spectra_KC10[0,-100]]) # (576,) array
z = modified_z_score(np.diff(y)) # (575,) array

threshold = 6 # binarization threshold. 
z_bool = np.abs(z) > threshold # (575,) array of booleans
plt.plot(x,y, "r-")
plt.plot(x,fixer(y,z_bool), 'b-')


#-----------------------------------------------------------------------------

#fixed_intensity_KC10 = dict()
#for a in range(0,105,5):
    #for b in range(-100,5,5):
            #intensity = np.array([column[1] for column in spectra_KC10[a,b]])
            #wavelength = np.array([column[0] for column in spectra_KC10[a,b]])
           # intensity_fixed = fixer(intensity, m=5)
           # fixed_intensity_KC10[a,b] = np.reshape(intensity_fixed, (576,))



#---------------------------------end timer-----------------------------------
end_time = time.process_time()
print("Script runtime:", str(end_time - start_time), "\bs")
# last runtime = 3.5s