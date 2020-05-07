#-----------------------------General Instructions----------------------------

# Filenames must be named as "x map.txt" where x is the material name.
# The filepath is C:\\Users\\Hector\\Desktop\\Data\\Map_Data\\ so this must
# be changed if the file is in a different location.

# This code produces a library of spectra with the keys being the materials'
# names, and within those keys a second key being the xy position. e.g. to 
# access KC10 spectra x=0,y=-100 then type: sliced_data["KC10"][(0,-100)]


#---------------------------------User Inputs---------------------------------

# step size in data in x (column 1) and y (column 2):
xstep = 5
ystep = 5


#-------------------------------Import Modules--------------------------------

import numpy as np
import time
import glob


#--------------------------------Import Data----------------------------------

# Start process timer
start_time = time.process_time() 

# Create a list of all files in filepath that have the form *.txt
filenames = sorted(glob.glob("C:\\Users\\Hector\\Desktop\\Data\\Map_Data\\*.txt"))

# Extract the material names from the filepath name
material = list()
for i in range(len(filenames)):
    material.append(filenames[i][38:-8]) # Corresponds to position of name

# This numpy method loads the text in as an array   
raw_data = dict()
for i in range(len(filenames)):
        raw_data[material[i]] = np.loadtxt(filenames[i])

# Slice data into individual spectra
sliced_data = dict()
for i in material:
    xmin = int(min(raw_data[i][:,0]))
    xmax = int(max(raw_data[i][:,0]) + xstep)
    ymin = int(min(raw_data[i][:,1]))
    ymax = int(max(raw_data[i][:,1]) + ystep)
    spectra = dict()
    for x in range(xmin, xmax, xstep):
        for y in range(ymin, ymax, ystep):
            posx = np.where(raw_data[i][:,0] == x) # coords when true
            posy = np.where(raw_data[i][:,1] == y) # coords when true
            inter = np.intersect1d(posx,posy) # intersection of posx & posy
            spectra[x,y] = raw_data[i][inter] # assign each spectrum to dict
    sliced_data[i] = spectra # for each material, embed spectra dict in dict

# End process timer
end_time = time.process_time()
print("Script runtime:", str(end_time - start_time), "s")

# last runtime = 8.7s

#---------------------------------Script End----------------------------------