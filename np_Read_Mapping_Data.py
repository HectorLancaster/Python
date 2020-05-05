import numpy as np
import time

# Starts a timer in seconds.
start_time = time.process_time()

# This np method reads in the txt data into an array.
KC10 = np.loadtxt("\\Users\\Hector\\Desktop\\Data\\KC10 map.txt")
# This splits the array, columnn wise, into 4 and assigns given variables.
x_KC10, y_KC10, wavenumber_KC10, intensity_KC10 = np.hsplit(KC10, 4)
    
LiC10 = np.loadtxt("\\Users\\Hector\\Desktop\\Data\\LiC10 map.txt")
x_LiC10, y_LiC10, wavenumber_LiC10, intensity_LiC10 = np.hsplit(LiC10, 4)
    
yp50 = np.loadtxt("\\Users\\Hector\\Desktop\\Data\\yp50 map.txt")
x_yp50, y_yp50, wavenumber_yp50, intensity_yp50 = np.hsplit(yp50, 4)



n_rows = KC10.shape[0]
spectra_KC10 = dict()
for a in range(0,105,5):            
    for b in range(-100,5,5):
        data = np.zeros((576,2))
        counter = 0
        for i in range(n_rows):
            if x_KC10[i] == a and y_KC10[i] == b:
                data[counter][0] += wavenumber_KC10[i]
                data[counter][1] += intensity_KC10[i]
                counter += 1
        spectra_KC10[a,b] = data


# This records the time at the end of the script.
end_time = time.process_time()
# Prints the runtime.
print("Script runtime:", str(end_time - start_time), "s")

# Last runtime: 166s
# Thus, if you want to import as arrays and sort to a dictionary, then 
# it takes a hell of a lot longer! (extrapolating, 8 times longer)