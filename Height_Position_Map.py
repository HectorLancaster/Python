# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 10:20:06 2020

@author: Hector
"""

# no way here of removing outliers, very important this is done otherwise
# the x coord coresponding to the max y could represent the x coord of the 
# outlier
import matplotlib.pyplot as plt
import numpy as np

plt.figure() 

#max_loc = np.zeros((21,21))
#for a in range(0,105,5):
    #for b in range(-100,5,5):
        #array_temp = np.array(spectra_KC10[a,b])
        #array_spectra = array_temp[:int(array_temp.shape[0]/2)]
        #max_loc_val = array_spectra[array_spectra.argmax(axis = 0)[1]][0]
        #max_loc[a//5,(b-5)//5] = max_loc_val

# Produces a 2D image from the input array, sets the aspect ratio to 'equal'
plt.subplot(223)
plt.imshow(max_loc_yp50, aspect='equal', cmap='Reds', interpolation='none')
plt.colorbar()
plt.yticks(np.arange(0,21, step=5))
plt.xticks(np.arange(0,21, step=5))
plt.title("yp50")

plt.subplot(221)
plt.imshow(max_loc_KC10, aspect='equal', cmap='Reds', interpolation='none')
#plt.colorbar()
plt.yticks([],[])
plt.xticks([],[])
plt.title("KC10")

plt.subplot(222)
plt.imshow(max_loc_LiC10, aspect='equal', cmap='Reds', interpolation='none')
#plt.colorbar()
plt.yticks([],[])
plt.xticks([],[])
plt.title("LiC10")

plt.savefig("C:\\Users\\Hector\\Desktop\\Data\\height position map.pdf")