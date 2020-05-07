#--------------------------------import modules-------------------------------
import numpy as np
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

# KC10 ------------------------
shape = list(np.array(spectra_KC10[0, -100]).shape)[0]
sum_y = np.zeros(shape) # initialise array for summing
x = np.array([column[0] for column in spectra_KC10[0,-100]]) # for plotting
KC10_clean = dict()
for a in range(0,105,5):            
    for b in range(-100,5,5):
        y = np.array([column[1] for column in spectra_KC10[a,b]]) # (1)
        z = modified_z_score(np.diff(y)) # (2)   
        z_bool = np.abs(z) > threshold # (3)
        y_fixed = fixer(y,z_bool) # send to fixer function
        KC10_clean[a,b] = [x,y_fixed]
        KC10_clean[a,b] = np.transpose(np.array(KC10_clean[a,b]))
        #KC10_clean[a,b] = np.reshape

# (1) create a (576,) array of intensity values y
# (2) send a (575,) array of differences in y to zscore function
# (3) create a (575,) array of booleans which give true if a spike is detected
# in the data, correspinding to a zscore greater than the threshold.


# LiC10 -----------------------
sum_y = np.zeros(shape)
x = np.array([column[0] for column in spectra_LiC10[0,-100]])
LiC10_clean = dict()
for a in range(0,105,5):            
    for b in range(-100,5,5):
        y = np.array([column[1] for column in spectra_LiC10[a,b]]) 
        z = modified_z_score(np.diff(y))      
        z_bool = np.abs(z) > threshold
        y_fixed = fixer(y,z_bool)
        LiC10_clean[a,b] = [x,y_fixed]
        LiC10_clean[a,b] = np.transpose(np.array(LiC10_clean[a,b]))

# yp50 ------------------------       
sum_y = np.zeros(shape)
yp50_clean = dict()
for a in range(0,105,5):            
    for b in range(-100,5,5):
        y = np.array([column[1] for column in spectra_yp50[a,b]])
        z = modified_z_score(np.diff(y))    
        z_bool = np.abs(z) > threshold 
        y_fixed = fixer(y,z_bool)
        yp50_clean[a,b] = [x,y_fixed]
        yp50_clean[a,b] = np.transpose(np.array(yp50_clean[a,b]))        


#-----------------------------------------------------------------------------

end_time = time.process_time()
print("Script runtime:", str(end_time - start_time), "s")
# last runtime = 0.4s