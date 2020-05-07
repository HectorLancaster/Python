#--------------------------------import modules-------------------------------
import numpy as np
import matplotlib.pyplot as plt
import time

#---------------------------------start timer---------------------------------
start_time = time.process_time()

#--------------------------------define zscore--------------------------------


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

threshold = 6 # binarization threshold. 

#-----------------------------------Plotting----------------------------------

plt.figure()

# KC10 ------------------------
shape = list(np.array(spectra_KC10[0, -100]).shape)[0] # number of data in a spectrum
sum_y = np.zeros(shape) # initialise array for summing
x = np.array([column[0] for column in spectra_KC10[0,-100]]) # for plotting
for a in range(0,105,5):            
    for b in range(-100,5,5):
        plt.subplot(221)
        y = np.array([column[1] for column in spectra_KC10[a,b]]) # (1)
        z = modified_z_score(np.diff(y)) # (2)   
        z_bool = np.abs(z) > threshold # (3)
        y_fixed = fixer(y,z_bool) # send to fixer function
        sum_y = sum_y + y_fixed # sum all fixed spectra
        max_g = max(y_fixed[:shape//2]) # find the max g peak per spectrum
        y_fixed_norm = y_fixed/max_g # normalise the spectra to the g peak
        plt.plot(x, y_fixed_norm, "b.", markersize = 1) # plot

# (1) create a (576,) array of intensity values y
# (2) send a (575,) array of differences in y to zscore function
# (3) create a (575,) array of booleans which give true if a spike is detected
# in the data, correspinding to a zscore greater than the threshold.

avspectra = sum_y/len(spectra_KC10) # calculate the average spectra
max_g = max(avspectra[:shape//2]) # find max g peak intensity for avspectra
norm_av = avspectra/max_g # normalise the average spectra
plt.plot(x, norm_av, "k-", linewidth = 1, label = "KC10 average") # plot

plt.yticks([],[])
plt.axis([1200, 1700, 0.2, 1.5])
plt.tick_params(axis='x', direction='in')
plt.legend(loc="upper left", fontsize="x-small", markerfirst=True,
           edgecolor="k", fancybox=False)


# LiC10 -----------------------
sum_y = np.zeros(shape)
x = np.array([column[0] for column in spectra_LiC10[0,-100]])
for a in range(0,105,5):            
    for b in range(-100,5,5):
        plt.subplot(222)
        y = np.array([column[1] for column in spectra_LiC10[a,b]]) 
        z = modified_z_score(np.diff(y))      
        z_bool = np.abs(z) > threshold
        y_fixed = fixer(y,z_bool)
        sum_y = sum_y + y_fixed
        max_g = max(y_fixed[:shape//2])
        y_fixed_norm = y_fixed/max_g
        plt.plot(x, y_fixed_norm, "g.", markersize = 1) 

LiC10_avspectra = sum_y/len(spectra_LiC10)
max_g = max(avspectra[:shape//2])
norm_av = avspectra/max_g
plt.plot(x, norm_av, "k-", linewidth = 1, label = "LiC10 average")


plt.yticks([],[])
plt.axis([1200, 1700, 0.2, 1.5])
plt.tick_params(axis='x', direction='in')
plt.legend(loc="upper left", fontsize="x-small", markerfirst=True,
           edgecolor="k", fancybox=False)


# yp50 ------------------------       
sum_y = np.zeros(shape)
x = np.array([column[0] for column in spectra_yp50[0,-100]])
for a in range(0,105,5):            
    for b in range(-100,5,5):
        plt.subplot(223)
        y = np.array([column[1] for column in spectra_yp50[a,b]])
        z = modified_z_score(np.diff(y))    
        z_bool = np.abs(z) > threshold 
        y_fixed = fixer(y,z_bool)
        sum_y = sum_y + y_fixed
        max_g = max(y_fixed[:shape//2])
        y_fixed_norm = y_fixed/max_g
        plt.plot(x, y_fixed_norm, "r.", markersize = 1) 
        
yp50_avspectra = sum_y/len(spectra_yp50)
max_g = max(avspectra[:shape//2])
norm_av = avspectra/max_g
plt.plot(x, norm_av, "k-", linewidth = 1, label = "yp50 average")


plt.xlabel("Raman shift (cm⁻¹)")
plt.ylabel("Intensity (arb. units)")
plt.yticks([],[])
plt.axis([1200, 1700, 0.2, 1.5])
plt.tick_params(axis='x', direction='in')
plt.legend(loc="upper left", fontsize="x-small", markerfirst=True,
           edgecolor="k", fancybox=False)

plt.savefig("C:\\Users\\Hector\\Desktop\\Data\\variation_in_spectra.pdf")

#-----------------------------------------------------------------------------

end_time = time.process_time()
print("Script runtime:", str(end_time - start_time), "s")
# last runtime = 24s