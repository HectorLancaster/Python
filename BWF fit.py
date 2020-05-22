#--------------------------------file notes-----------------------------------

# Applies a Lorentzian fit to the D peak and Breit-Wigner-Fano fit to the G 
# peak. 

# To do:
#       1) Include baseline as variable (see Chris' code)


#--------------------------------user inputs----------------------------------



#------------------------------import modules---------------------------------

import time
import numpy as np
import scipy
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.ticker as ticker


#----------------------------start process timer------------------------------

start_time = time.process_time()

#-----------------------------define functions--------------------------------

# See cauchy distribution function at:
# https://www.itl.nist.gov/div898/handbook/eda/section3/eda3663.htm 
# Here, the  modified version with the max intensity (I0D) replaces the 
# the parameter (1/s*pi).
def Lorentzian(x, I0D, cenD, gammaD):
    return I0D*gammaD**2/((x-cenD)**2+gammaD**2)

# Function as described in SI of DOI: 10.1103/PhysRevB.84.241404    
def BWF(x, I0G, gammaG, cenG, q):
    numerator = (1+(x-cenG)/(q*gammaG))**2
    denominator = 1+((x-cenG)/(gammaG))**2
    return I0G*(numerator/denominator)

# Combined function of above two.
def LorBWF(x, I0D, cenD, gammaD, I0G, gammaG, cenG, q):
    numerator = (1+(x-cenG)/(q*gammaG))**2
    denominator = 1+((x-cenG)/(gammaG))**2
    return I0D*gammaD**2/((x-cenD)**2+gammaD**2) + I0G*(numerator/denominator)

        

#--------------------------get & print fitting data---------------------------

for i in material: # for each material
    #----standard----
    xmin = int(min(raw_data[i][:,0]))
    ymin = int(min(raw_data[i][:,1]))
    #----------------
    x = norm_data[i][xmin,ymin][:,2] # define the spectral range
    y = av_data[i] # use the map average as the y coords
    baseline = min(y) # use minimum value as baseline
    y = y - baseline # subtract from average data
    
    #-----------------------------initial guesses-----------------------------    

    #--Lorentzian D-peak---
    I0D = max(y) # magnitude of the D peak
    cenD = x[y.argmax(axis=0)] # central frequency of D peak
    gammaD = 40 # half width at half maximum (HWHM)
    
    #--BWF G-peak---
    I0G = max(y[:288]) # magnitude of the G peak
    gammaG = 60 # half width at half maximum (HWHM)
    cenG = x[y[:288].argmax(axis=0)] # central frequency of G peak
    q = -5 # where 1/q is the Fano parameter

                
    #-------------------------------find voigt fit----------------------------   
    
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

    popt_LorBWF, pcov_LorBWF = scipy.optimize.curve_fit(LorBWF,
                                                            x,
                                                            y,
                                                            p0=[I0D, cenD,
                                                                gammaD, I0G,
                                                                gammaG, cenG,
                                                                q])
    
    # this calculates the one standard deviation errors of the parameters
    # since var = sigma^2
    perr_LorBWF = np.sqrt(np.diag(pcov_LorBWF))
    
    # seperate parameters for each peak
    pars_1 = popt_LorBWF[0:3]
    pars_2 = popt_LorBWF[3:7]
    
    # define individual peaks based on individual parameters
    Lor_peak = Lorentzian(x, *pars_1)
    BWF_peak = BWF(x, *pars_2)
                  

    #---------------------------------residuals-------------------------------
    
    # calculate the residuals (difference between fit and data points)
    residual = y - (LorBWF(x, *popt_LorBWF))
    
    normres = residual/np.sqrt(y)
    num1 = np.where(normres > -1)
    num2 = np.where(normres < 1)
    inter = np.intersect1d(num1, num2)
    bound1 = (len(inter)/len(normres))*100
    num1 = np.where(normres > -2)
    num2 = np.where(normres < 2)
    inter = np.intersect1d(num1, num2)
    bound2 = (len(inter)/len(normres))*100    
    
    chi2 = sum(normres**2)
    
    
    #----------------------------------print----------------------------------
    
    # this cell prints the fitting parameters with their errors
    print("\n--" + i + "--")
    print("_____________D peak______________")
    print("\namplitude = %0.4f (+/-) %0.4f" % (pars_1[0], perr_LorBWF[0]))
    print("center = %0.2f (+/-) %0.2f" % (pars_1[1], perr_LorBWF[1]))
    print("HWHM = %0.2f (+/-) %0.2f" % (pars_1[2], perr_LorBWF[2]))
    print("area = %0.2f" % np.trapz(Lor_peak))
    print("_________________________________")
    print("\n_____________G peak______________")
    print("\namplitude = %0.4f (+/-) %0.4f" % (pars_2[0], perr_LorBWF[3]))
    print("HWHM = %0.2f (+/-) %0.2f" % (pars_2[1], perr_LorBWF[4]))
    print("center = %0.2f (+/-) %0.2f" % (pars_2[2], perr_LorBWF[5]))
    print("q = %0.2f (+/-) %0.2f" % (pars_2[3], perr_LorBWF[6]))
    print("area = %0.2f" % np.trapz(BWF_peak))
    print("_________________________________")
    print("\nnormalised res. between -1 and 1: %0.2f percent" % (bound1))
    print("normalised res. between -2 and 2: %0.2f percent" % (bound2))
    print("chi2: %0.2f" % (chi2))    
    
    
    #---------------------------------plotting--------------------------------
    # size in inches
    fig = plt.figure(figsize=(4,4))
    # specifies the geometry a grid that a subplot can be placed in
    gs = gridspec.GridSpec(2,1, height_ratios=[1,0.25])
    # adds subplots to the grid location specified
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    # updates the current values of the grid
    gs.update(hspace=0) 
    
    # average raman data
    ax1.plot(x, y, "o", markersize=1, color="gray", label="average data")
    
    # peak 1
    ax1.plot(x, Lor_peak, color="orange", alpha=0.5)
    ax1.fill_between(x, Lor_peak.min(), Lor_peak,
                     facecolor="orange", alpha=0.5)
      
    # peak 2
    ax1.plot(x, BWF_peak, color="cornflowerblue", alpha=0.5)
    ax1.fill_between(x, BWF_peak.min(), BWF_peak,
                     facecolor="cornflowerblue", alpha=0.5)  
    
    # fit: scipy.optimize.curve_fit
    ax1.plot(x, LorBWF(x, *popt_LorBWF), 'k--', label= "curvefit")
    
    # residuals
    ax2.plot(x, residual, "o", markersize = 1, color = "gray")              
    
    # plot specifications
    ax2.xaxis.set_minor_locator(AutoMinorLocator(4)) # number of minor ticks
    ax2.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(100))
    #ax1.xaxis.set_major_formatter(plt.NullFormatter())
    
    ax2.set_xlabel('Raman shift (cm⁻¹)')
    ax1.set_ylabel('Intensity (arb. units)', labelpad=11)
    ax2.set_ylabel('Res.', labelpad = 0.2)
    ax1.legend(loc="best", fontsize=8)
    #ax1.set_yticks([],[])
    
    ax1.set_ylim(-0.05, (max(y)+0.15))
    
    ax2.tick_params(axis ='x', direction ='in', which = "major", 
                    length=4, labelsize=8)
    ax2.tick_params(axis ='x', direction ='in', which = "minor", 
                    length=2,)
    ax2.tick_params(axis ='y', direction ='in', which = "major",
                    length=4, labelsize=8,
                    pad=5)
    ax2.tick_params(axis ='y', direction ='in', which = "minor",
                    length=2)
    #ax2.yaxis.tick_right()
    
    ax1.set_title(i, loc='left', pad = 0)
    fig.tight_layout()
    fig.savefig("C:\\Users\\Hector\\Desktop\\Data\\Figures\\" + i + " fitted.pdf")
    
    
#-----------------------------------------------------------------------------

# End process timer
end_time = time.process_time()
print("\nScript runtime: %.2f \bs" % (end_time - start_time))

# last runtime = 1.03s


#---------------------------------Script End----------------------------------              
