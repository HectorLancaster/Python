
# no way here of removing outliers, very important this is done otherwise
# the x coord coresponding to the max y could represent the x coord of the 
# outlier
import matplotlib.pyplot as plt
import numpy as np

max_loc = np.zeros((21,21))
for a in range(0,105,5):
    for b in range(-100,5,5):
        array_temp = np.array(spectra_KC10[a,b])
        array_spectra = array_temp[:int(array_temp.shape[0]/2)]
        max_loc_val = array_spectra[array_spectra.argmax(axis = 0)[1]][0]
        max_loc[a//5,(b-5)//5] = max_loc_val


plt.hist(max_loc, bins = 10, density = True);
plt.savefig("hist.pdf")


# Sample mean, varience and standard dev
print("Sample mean: ", str(max_loc.mean()))
print("Sample variance: ", str(max_loc.var()))
print("Sample standard deviation: ", str(max_loc.std()))