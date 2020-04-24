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

KC10_maxG = max([column[1] for column in \
                 KC10_avspectra[:int(KC10_avspectra.shape[0]/2)]])
norm_KC10 = [column[1]/KC10_maxG for column in KC10_avspectra]

LiC10_maxG = max([column[1] for column in \
                  LiC10_avspectra[:int(LiC10_avspectra.shape[0]/2)]])
norm_LiC10 = [column[1]/LiC10_maxG for column in LiC10_avspectra]

yp50_maxG = max([column[1] for column in \
                 yp50_avspectra[:int(yp50_avspectra.shape[0]/2)]])
norm_yp50 = [column[1]/yp50_maxG for column in yp50_avspectra]


plt.plot([column[0] for column in yp50_avspectra],
         norm_yp50,
         "r-", linewidth=1, label = "Raw")

plt.plot([column[0] for column in LiC10_avspectra],
         norm_LiC10,
         "g-", linewidth=1, label = "LiC10")

plt.plot([column[0] for column in KC10_avspectra],
         norm_KC10,
         "b-", linewidth=1, label = "KC10")


plt.xlabel("Raman shift (cm⁻¹)")
plt.ylabel("Intensity (arb. units)")

plt.yticks([],[])
plt.tick_params(axis='x', direction='in')

plt.axis([1200, 1700, 0.3, 1.3])

plt.legend(loc="upper left", fontsize="x-small", markerfirst=True,
           edgecolor="k", fancybox=False)

plt.savefig("composite average map (G normalised).pdf")


#-----------------------------Data Print--------------------------------------
print("\nMaximum intensity of yp50 is found at: " + \
      str(int(yp50_avspectra[yp50_avspectra.argmax(axis = 0)[1]][0])) + \
          " cm⁻¹")
RHS = yp50_avspectra[:int(yp50_avspectra.shape[0]/2)]
print("\tand the second maximum at: " + \
      str(int(yp50_avspectra[RHS.argmax(axis=0)[1]][0])) + " cm⁻¹")

print("\nMaximum intensity of LiC10 is found at: " + \
      str(int(LiC10_avspectra[LiC10_avspectra.argmax(axis = 0)[1]][0])) + \
          " cm⁻¹")
RHS = LiC10_avspectra[:int(LiC10_avspectra.shape[0]/2)]
print("\tand the second maximum at: " + \
      str(int(LiC10_avspectra[RHS.argmax(axis=0)[1]][0])) + " cm⁻¹")
        
print("\nMaximum intensity of KC10 is found at: " + \
      str(int(KC10_avspectra[KC10_avspectra.argmax(axis = 0)[1]][0])) + \
          " cm⁻¹")
RHS = KC10_avspectra[:int(KC10_avspectra.shape[0]/2)]
print("\tand the second maximum at: " + \
      str(int(KC10_avspectra[RHS.argmax(axis=0)[1]][0])) + " cm⁻¹")
