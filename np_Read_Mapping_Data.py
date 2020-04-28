import numpy as np
import time

start_time = time.process_time()

KC10 = \
    np.loadtxt("\\Users\\Hector\\Desktop\\Data\\KC10 map.txt", unpack = False)
x_KC10, y_KC10, wavenumber_KC10, intensity_KC10 = np.hsplit(KC10, 4)
    
LiC10 = \
    np.loadtxt("\\Users\\Hector\\Desktop\\Data\\LiC10 map.txt", unpack = False)
x_LiC10, y_LiC10, wavenumber_LiC10, intensity_LiC10 = np.hsplit(LiC10, 4)
    
yp50 = \
    np.loadtxt("\\Users\\Hector\\Desktop\\Data\\yp50 map.txt", unpack = False)
x_yp50, y_yp50, wavenumber_yp50, intensity_yp50 = np.hsplit(yp50, 4)

end_time = time.process_time()
print("Script runtime:", str(end_time - start_time), "s")

# Last runtime: 6.7s
# That's ~ 7X faster than original code "Read_Mapping_Data"