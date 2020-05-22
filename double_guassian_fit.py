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

x = norm_data["KC10"][0,-100][:,2]
y = av_data["KC10"]
y_line = y - y[0]
x_array = x
y_array_2gauss = y_line

plt.plot(x_array,y_array_2gauss, "bo", markersize = 1)

#-----------------------------define functions--------------------------------

def _2gaussian(x, amp1,cen1,sigma1, amp2,cen2,sigma2):
    return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x_array-cen1)/sigma1)**2))) + \
            amp2*(1/(sigma2*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x_array-cen2)/sigma2)**2)))

def _1gaussian(x, amp1,cen1,sigma1):
    return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x_array-cen1)/sigma1)**2)))


#-----------------------------initial guesses---------------------------------

amp1 = max(y_array_2gauss)
cen1 = x_array[y_array_2gauss.argmax(axis=0)]
#sigma1 = np.std(y_array_gauss, ddof=1) # ddof=1 mean N-1 on denominator i.e. sample
sigma1 = 40

amp2 = max(y_array_2gauss[:288])
cen2 = x_array[y_array_2gauss[:288].argmax(axis=0)]
sigma2 = 30


#----------------------------find guassian fit--------------------------------

popt_2gauss, pcov_2gauss = scipy.optimize.curve_fit(_2gaussian,
                                                    x_array,
                                                    y_array_2gauss,
                                                    p0=[amp1, cen1, sigma1,
                                                        amp2, cen2, sigma2],
                                                        bounds = (0,np.inf))
perr_2gauss = np.sqrt(np.diag(pcov_2gauss))
pars_1 = popt_2gauss[3:6]
pars_2 = popt_2gauss[0:3]
gauss_peak_1 = _1gaussian(x_array, *pars_1)
gauss_peak_2 = _1gaussian(x_array, *pars_2)

fig = plt.figure(figsize=(4,3))
gs = gridspec.GridSpec(1,1)
ax1 = fig.add_subplot(gs[0])

ax1.plot(x_array, y_array_2gauss, "bo", markersize = 1)
ax1.plot(x_array, _2gaussian(x_array, *popt_2gauss), 'k-.')


ax1.plot(x_array, gauss_peak_2, "y")
ax1.fill_between(x_array, gauss_peak_2.min(), gauss_peak_2, facecolor="yellow", alpha=0.5)  

ax1.plot(x_array, gauss_peak_1, "g")
ax1.fill_between(x_array, gauss_peak_1.min(), gauss_peak_1, facecolor="green", alpha=0.5)
  

#----------------------------------print--------------------------------------

# note: need to double check that the peak1/2 data is indexed right for pars &
#       perr

# this cell prints the fitting parameters with their errors
print("\n-------------Peak 1-------------")
print("amplitude = %0.2f (+/-) %0.2f" % (pars_1[0], perr_2gauss[3]))
print("center = %0.2f (+/-) %0.2f" % (pars_1[1], perr_2gauss[4]))
print("sigma = %0.2f (+/-) %0.2f" % (pars_1[2], perr_2gauss[5]))
print("area = %0.2f" % np.trapz(gauss_peak_1))
print("--------------------------------")
print("-------------Peak 2-------------")
print("amplitude = %0.2f (+/-) %0.2f" % (pars_2[0], perr_2gauss[0]))
print("center = %0.2f (+/-) %0.2f" % (pars_2[1], perr_2gauss[1]))
print("sigma = %0.2f (+/-) %0.2f" % (pars_2[2], perr_2gauss[2]))
print("area = %0.2f" % np.trapz(gauss_peak_2))
print("--------------------------------")


#---------------------------------residuals-----------------------------------

residual_2gauss = y_array_2gauss - (_2gaussian(x_array, *popt_2gauss))

fig = plt.figure(figsize=(4,4))
gs = gridspec.GridSpec(2,1, height_ratios=[1,0.25])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
gs.update(hspace=0) 

ax1.plot(x_array, y_array_2gauss, "ro", markersize = 1)
ax1.plot(x_array, _2gaussian(x_array, *popt_2gauss), 'k--')#,\
         #label="y= %0.2f$e^{%0.2fx}$ + %0.2f" % (popt_exponential[0], popt_exponential[1], popt_exponential[2]))

# peak 1
ax1.plot(x_array, gauss_peak_1, "g")
ax1.fill_between(x_array, gauss_peak_1.min(), gauss_peak_1, facecolor="green", alpha=0.5)
  
# peak 2
ax1.plot(x_array, gauss_peak_2, "y")
ax1.fill_between(x_array, gauss_peak_2.min(), gauss_peak_2, facecolor="yellow", alpha=0.5)  

# residual
ax2.plot(x_array, residual_2gauss, "bo", markersize = 1)

#-----------------------------------------------------------------------------

# End process timer
end_time = time.process_time()
print("Script runtime: %.2f \bs" % (end_time - start_time))

# last runtime = 0.38s


#---------------------------------Script End----------------------------------