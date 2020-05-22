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
y_line = y - y[575]
x_array = x[288:]
y_array_gauss = y_line[288:]

plt.plot(x_array,y_array_gauss, "bo", markersize = 1)


#-----------------------------define functions--------------------------------

def _1gaussian(x, amp1,cen1,sigma1):
    return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x_array-cen1)/sigma1)**2)))


#-----------------------------initial guesses---------------------------------

amp1 = max(y_array_gauss)
cen1 = x_array[y_array_gauss.argmax(axis=0)]
#sigma1 = np.std(y_array_gauss, ddof=1) # ddof=1 mean N-1 on denominator i.e. sample
sigma1 = 40


#----------------------------find guassian fit--------------------------------

# The below is taken from the documentation on scipy.org:
# scipy.optimize.curve_fit(f, xdata, ydata, p0)
# uses a non-linear least squares to fit a function, f, to data.
# f = the model function
# xdata = the independent variable where the data is measured
# ydata = the dependent data
# p0 = initial guess for the parameters of f

# returns: 
#   popt: optimal values for the parameters so that the sum of the squared
#         residuals of f(xdata, *popt) - ydata is minimized
#   pcov: the estimated covarience of popt. The diagonals provide the variance
#         of the parameter estimate.

popt_gauss, pcov_gauss = scipy.optimize.curve_fit(_1gaussian, x_array, 
                                                  y_array_gauss,
                                                  p0=[amp1, cen1, sigma1],
                                                  bounds = (0,np.inf))

# this calculates the one standard deviation errors of the parameters
perr_gauss = np.sqrt(np.diag(pcov_gauss))


print("amplitude = %0.2f (+/-) %0.2f" % (popt_gauss[0], perr_gauss[0]))
print("center = %0.2f (+/-) %0.2f" % (popt_gauss[1], perr_gauss[1]))
print("sigma = %0.2f (+/-) %0.2f" % (popt_gauss[2], perr_gauss[2]))


fig = plt.figure(figsize=(4,3))
gs = gridspec.GridSpec(1,1)
ax1 = fig.add_subplot(gs[0])

ax1.plot(x_array, y_array_gauss, "bo", markersize =1) # experimental data
ax1.plot(x_array, _1gaussian(x_array, *popt_gauss), 'k-.')  # fit data, (1)

# (1) here, the *popt_gauss passess all the arguments of popt_gauss to the
#     function _1gaussian, instead of writing them all out explicitly.

#-----------------------------------------------------------------------------

# End process timer
end_time = time.process_time()
print("Script runtime: %.2f \bs" % (end_time - start_time))

# last runtime = 0.08s


#---------------------------------Script End----------------------------------








