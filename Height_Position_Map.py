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

max_loc = np.zeros((21,21))
for a in range(0,105,5):
    for b in range(-100,5,5):
        array_temp = np.array(spectra_KC10[a,b])
        array_spectra = array_temp[:int(array_temp.shape[0]/2)]
        max_loc_val = array_spectra[array_spectra.argmax(axis = 0)[1]][0]
        max_loc[a//5,(b-5)//5] = max_loc_val

# Produces a 2D image from the input array, sets the aspect ratio to 'equal'
plt.imshow(max_loc, aspect='equal') 

plt.xlabel("x-pos.")
plt.ylabel("y-pos.")


plt.xticks([],[])
plt.yticks([],[])

#plt.hist(max_loc, bins = 30);
#plt.savefig("hist.pdf")