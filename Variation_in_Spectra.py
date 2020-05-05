import matplotlib.pyplot as plt
import numpy as np
import time

start_time = time.process_time()

#------------------------------Averages---------------------------------------
# Find the shape of an individual spectra for the given key,
# this returns a tuple, but we need a list for later operations.
spectra_shape = list(np.array(spectra_KC10[0, -100]).shape)
# Get the length of the dictionary, to know how many spectra there are. 
length_dict = len(spectra_KC10)
# Create an empty array of zeros of the shape of a given spectra.
sum_array = np.zeros(spectra_shape)
# This loop sums all array enteries to give one combined summation spectra.
for a in range(0,105,5):
    for b in range(-100,5,5):
        # This adds the values of each element of the arrays.
        sum_array = sum_array + np.array(spectra_KC10[a,b])
# divide the summed spectra by the number of enteries.         
KC10_avspectra = sum_array/(length_dict)

spectra_shape = list(np.array(spectra_LiC10[0, -100]).shape)
length_dict = len(spectra_LiC10)
sum_array = np.zeros(spectra_shape)
for a in range(0,105,5):
    for b in range(-100,5,5):
        sum_array = sum_array + np.array(spectra_LiC10[a,b])
LiC10_avspectra = sum_array/(length_dict)

spectra_shape = list(np.array(spectra_yp50[0, -100]).shape)
length_dict = len(spectra_yp50)
sum_array = np.zeros(spectra_shape)
for a in range(0,105,5):
    for b in range(-100,5,5):
        sum_array = sum_array + np.array(spectra_yp50[a,b])
yp50_avspectra = sum_array/(length_dict)

#------------------------------Plotting---------------------------------------
plt.figure()
for a in range(0,105,5):            
    for b in range(-100,5,5):
        plt.subplot(221)
        x = [column[0] for column in spectra_KC10[a,b]]
        y = [column[1] for column in spectra_KC10[a,b]]
        
        plt.plot(x, y, "b.")

plt.subplot(221)
x = [column[0] for column in KC10_avspectra]
y = [column[1] for column in KC10_avspectra]
plt.plot(x, y, "k-", linewidth=2, label = "KC10 Average")

plt.yticks([],[])
#plt.xticks(np.arange(1200,1700,step = 150))
plt.tick_params(axis='x', direction='in')
plt.legend(loc="upper left", fontsize="x-small", markerfirst=True,
           edgecolor="k", fancybox=False)
plt.axis([1200, 1700, 0, 1000])


for a in range(0,105,5):            
    for b in range(-100,5,5):
        plt.subplot(222)
       
        y = [column[1] for column in spectra_LiC10[a,b]]
        plt.plot(x, y, "g.")

plt.subplot(222)        
x = [column[0] for column in LiC10_avspectra]
y = [column[1] for column in LiC10_avspectra]
plt.plot(x, y, "k-", linewidth=2, label = "LiC10 Average")

plt.yticks([],[])
#plt.xticks(np.arange(1200,1700, step = 1500))
plt.tick_params(axis='x', direction='in')
plt.legend(loc="upper left", fontsize="x-small", markerfirst=True,
           edgecolor="k", fancybox=False)
plt.axis([1200, 1700, 0, 2500])

       
for a in range(0,105,5):            
    for b in range(-100,5,5):
        plt.subplot(223)
        x = [column[0] for column in spectra_yp50[a,b]]
        y = [column[1] for column in spectra_yp50[a,b]]
        plt.plot(x, y, "r.")

plt.subplot(223)
x = [column[0] for column in yp50_avspectra]
y = [column[1] for column in yp50_avspectra]
plt.plot(x, y, "k-", linewidth=2, label = "Raw Average")


plt.xlabel("Raman shift (cm⁻¹)")
plt.ylabel("Intensity (arb. units)")
plt.yticks([],[])
#plt.xticks(np.arange(1200,1700, step = 1500))
plt.tick_params(axis='x', direction='in')
plt.legend(loc="upper left", fontsize="x-small", markerfirst=True,
           edgecolor="k", fancybox=False)
plt.axis([1200, 1700, 0, 1100])

plt.savefig("variation in spectra.pdf")

#-----------------------------------------------------------------------------

end_time = time.process_time()
print("Script runtime:", str(end_time - start_time), "s")
# last runtime = 26.5s