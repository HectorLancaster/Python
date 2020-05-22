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
x = x[:288]
y = y_line[:288]

plt.plot(x, y, "bo", markersize = 1)


#-----------------------------define functions--------------------------------


def BWF(x, I0G, gammaG, cenG, q):
    numerator = (1+(x-cenG)/(q*gammaG))**2
    denominator = 1+((x-cenG)/(gammaG))**2
    return I0G*(numerator/denominator)

#-----------------------------initial guesses---------------------------------

I0G = max(y)
cenG = x[y.argmax(axis=0)]
gammaG = 40
q = -5


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

popt, pcov = scipy.optimize.curve_fit(BWF, x, y, p0=[I0G, gammaG, cenG, q],
                                                  bounds = (-np.inf,np.inf))
                                                  

# this calculates the one standard deviation errors of the parameters
perr = np.sqrt(np.diag(pcov))


print("I0G = %0.2f (+/-) %0.2f" % (popt[0], perr[0]))
print("HWHM = %0.2f (+/-) %0.2f" % (popt[1], perr[1]))
print("Center G = %0.2f (+/-) %0.2f" % (popt[2], perr[2]))
print("q = %0.2f (+/-) %0.2f" % (popt[3], perr[3]))


fig = plt.figure(figsize=(4,3))
gs = gridspec.GridSpec(1,1)
ax1 = fig.add_subplot(gs[0])

ax1.plot(x, y, "bo", markersize =1) # experimental data
ax1.plot(x, BWF(x, *popt), 'k-.')  # fit data, (1)

# (1) here, the *popt_gauss passess all the arguments of popt_gauss to the
#     function _1gaussian, instead of writing them all out explicitly.

#-----------------------------------------------------------------------------

# End process timer
end_time = time.process_time()
print("Script runtime: %.2f \bs" % (end_time - start_time))

# last runtime = 0.08s


#---------------------------------Script End----------------------------------








