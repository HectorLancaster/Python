#--------------------------------file notes-----------------------------------

# Applies a Gaussian fit to the data


#--------------------------------user inputs----------------------------------


#------------------------------import modules---------------------------------

import time
import numpy as np
import scipy
from matplotlib import gridspec
import matplotlib.pyplot as plt


#----------------------------start process timer------------------------------

start_time = time.process_time()


#--------------------------get & print fitting data---------------------------

x = norm_data["yp50"][0,-100][:,2]
y = av_data["yp50"]
y_line = y - y[0]
x_array = x
y_array_2lorentz = y_line

plt.plot(x_array,y_array_2lorentz, "bo", markersize = 1)

#-----------------------------define functions--------------------------------

def _1Lorentzian(x, amp, cen, wid):
    return amp*wid**2/((x-cen)**2+wid**2)

def _2Lorentzian(x, amp1, cen1, wid1, amp2,cen2,wid2):
    return (amp1*wid1**2/((x-cen1)**2+wid1**2)) +\
            (amp2*wid2**2/((x-cen2)**2+wid2**2))

#-----------------------------initial guesses---------------------------------

amp1 = max(y_array_2lorentz)
cen1 = x_array[y_array_2lorentz.argmax(axis=0)]
#sigma1 = np.std(y_array_gauss, ddof=1) # ddof=1 mean N-1 on denominator i.e. sample
wid1 = 40

amp2 = max(y_array_2lorentz[:288])
cen2 = x_array[y_array_2lorentz[:288].argmax(axis=0)]
wid2 = 30


#----------------------------find guassian fit--------------------------------

popt_2lorentz, pcov_2lorentz = scipy.optimize.curve_fit(_2Lorentzian,
                                                        x_array,
                                                        y_array_2lorentz,
                                                        p0=[amp1, cen1, wid1,
                                                            amp2, cen2, wid2],
                                                        bounds = (0,np.inf))

perr_2lorentz = np.sqrt(np.diag(pcov_2lorentz))

pars_1 = popt_2lorentz[3:6]
pars_2 = popt_2lorentz[0:3]

lorentz_peak_1 = _1Lorentzian(x_array, *pars_1)
lorentz_peak_2 = _1Lorentzian(x_array, *pars_2)

#----------------------------------print--------------------------------------

# note: need to double check that the peak1/2 data is indexed right for pars &
#       perr

# this cell prints the fitting parameters with their errors
print("\n-------------Peak 1-------------")
print("amplitude = %0.2f (+/-) %0.2f" % (pars_1[0], perr_2lorentz[3]))
print("center = %0.2f (+/-) %0.2f" % (pars_1[1], perr_2lorentz[4]))
print("width = %0.2f (+/-) %0.2f" % (pars_1[2], perr_2lorentz[5]))
print("area = %0.2f" % np.trapz(lorentz_peak_1))
print("--------------------------------")
print("-------------Peak 2-------------")
print("amplitude = %0.2f (+/-) %0.2f" % (pars_2[0], perr_2lorentz[0]))
print("center = %0.2f (+/-) %0.2f" % (pars_2[1], perr_2lorentz[1]))
print("width = %0.2f (+/-) %0.2f" % (pars_2[2], perr_2lorentz[2]))
print("area = %0.2f" % np.trapz(lorentz_peak_2))

#---------------------------------residuals-----------------------------------

residual_2lorentz = y_array_2lorentz - (_2Lorentzian(x_array, *popt_2lorentz))


fig = plt.figure(figsize=(4,4))
gs = gridspec.GridSpec(2,1, height_ratios=[1,0.25])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
gs.update(hspace=0) 

ax1.plot(x_array, y_array_2lorentz, "ro", markersize = 1)
ax1.plot(x_array, _2Lorentzian(x_array, *popt_2lorentz), 'k--')#,\
         #label="y= %0.2f$e^{%0.2fx}$ + %0.2f" % (popt_exponential[0], popt_exponential[1], popt_exponential[2]))

# peak 1
ax1.plot(x_array, lorentz_peak_1, "g")
ax1.fill_between(x_array, lorentz_peak_1.min(), lorentz_peak_1, facecolor="green", alpha=0.5)
  
# peak 2
ax1.plot(x_array, lorentz_peak_2, "y")
ax1.fill_between(x_array, lorentz_peak_2.min(), lorentz_peak_2, facecolor="yellow", alpha=0.5)  


# residual
ax2.plot(x_array, residual_2lorentz, "bo", markersize = 1)

#-----------------------------------------------------------------------------

# End process timer
end_time = time.process_time()
print("Script runtime: %.2f \bs" % (end_time - start_time))

# last runtime = 0.14s


#---------------------------------Script End----------------------------------