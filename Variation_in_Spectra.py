import matplotlib.pyplot as plt
import numpy as np

#------------------------------Averages---------------------------------------
spectra_shape = list(np.array(spectra_KC10[0, -100]).shape)
length_dict = len(spectra_KC10)
sum_array = np.zeros(spectra_shape)
for a in range(0,105,5):
    for b in range(-100,5,5):
        sum_array = sum_array + np.array(spectra_KC10[a,b])
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
        plt.plot([column[0] for column in spectra_KC10[a,b]],
                 [column[1] for column in spectra_KC10[a,b]],
                 "b.")

plt.subplot(221)
plt.plot([column[0] for column in KC10_avspectra],
         [column[1] for column in KC10_avspectra],
         "k-", linewidth=2, label = "KC10 Average")

plt.yticks([],[])
plt.tick_params(axis='x', direction='in')

plt.legend(loc="upper left", fontsize="x-small", markerfirst=True,
           edgecolor="k", fancybox=False)

plt.axis([1200, 1700, 0, 1000])


for a in range(0,105,5):            
    for b in range(-100,5,5):
        plt.subplot(222)
        plt.plot([column[0] for column in spectra_LiC10[a,b]],
                 [column[1] for column in spectra_LiC10[a,b]],
                 "g.")

plt.subplot(222)        
plt.plot([column[0] for column in LiC10_avspectra],
         [column[1] for column in LiC10_avspectra],
         "k-", linewidth=2, label = "LiC10 Average")

plt.yticks([],[])
plt.tick_params(axis='x', direction='in')

plt.legend(loc="upper left", fontsize="x-small", markerfirst=True,
           edgecolor="k", fancybox=False)

plt.axis([1200, 1700, 0, 2500])

       
for a in range(0,105,5):            
    for b in range(-100,5,5):
        plt.subplot(223)
        plt.plot([column[0] for column in spectra_yp50[a,b]],
                 [column[1] for column in spectra_yp50[a,b]],
                 "r.")

plt.subplot(223)
plt.plot([column[0] for column in yp50_avspectra],
         [column[1] for column in yp50_avspectra],
         "k-", linewidth=2, label = "Raw Average")


plt.xlabel("Raman shift (cm⁻¹)")
plt.ylabel("Intensity (arb. units)")

plt.yticks([],[])
plt.tick_params(axis='x', direction='in')

plt.legend(loc="upper left", fontsize="x-small", markerfirst=True,
           edgecolor="k", fancybox=False)

plt.axis([1200, 1700, 0, 1100])

plt.savefig("variation in spectra.pdf")

