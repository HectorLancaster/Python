#--------------------------------file notes-----------------------------------

# Applies a Lorentzian fit to the D peak and Breit-Wigner-Fano fit to the G 
# peak. 

# To do:
#       1) Include baseline as variable (see Chris' code)


#--------------------------------user inputs----------------------------------

# Step size in data in x (column 1) and y (column 2):
xstep = 5
ystep = 5

#------------------------------import modules---------------------------------

import time
import numpy as np
import scipy
import pandas as pd

#----------------------------start process timer------------------------------

start_time = time.process_time()

#-----------------------------define functions--------------------------------

# See cauchy distribution function at:
# https://www.itl.nist.gov/div898/handbook/eda/section3/eda3663.htm 
# Here, the  modified version with the max intensity (I0D) replaces the 
# the parameter (1/s*pi).
def Lorentzian(x, I0D, cenD, gammaD, background):
    return I0D*gammaD**2/((x-cenD)**2+gammaD**2) + background

# Function as described in SI of DOI: 10.1103/PhysRevB.84.241404    
def BWF(x, I0G, gammaG, cenG, q, background):
    numerator = (1+(x-cenG)/(q*gammaG))**2
    denominator = 1+((x-cenG)/(gammaG))**2
    return I0G*(numerator/denominator) + background

# Combined function of above two.
def LorBWF(x, I0D, cenD, gammaD, I0G, gammaG, cenG, q, background):
    numerator = (1+(x-cenG)/(q*gammaG))**2
    denominator = 1+((x-cenG)/(gammaG))**2
    return I0D*gammaD**2/((x-cenD)**2+gammaD**2) +\
        I0G*(numerator/denominator) + background

        

#--------------------------get & print fitting data---------------------------

for i in material: # for each material
    xmin = int(min(raw_data[i][:,0]))
    xmax = int(max(raw_data[i][:,0]) + xstep)
    ymin = int(min(raw_data[i][:,1]))
    ymax = int(max(raw_data[i][:,1]) + ystep)
    spectra = dict()
    for x in range(xmin, xmax, xstep):
        for y in range(ymin, ymax, ystep):
            #----standard----
            xmin = int(min(raw_data[i][:,0]))
            ymin = int(min(raw_data[i][:,1]))
            #----------------
            xs = clean_data[i][x,y][:,2]
            ys = clean_data[i][x,y][:,3]
            
            #--------------------------initial guesses----------------------------    
        
            #--Lorentzian D-peak---
            I0D = max(ys[288:576]) # magnitude of the D peak
            cenD = xs[np.where(ys==I0D)] # central frequency of D peak
            gammaD = 50 # half width at half maximum (HWHM)
            
            #--BWF G-peak---
            I0G = max(ys[:288]) # magnitude of the G peak
            gammaG = 30 # half width at half maximum (HWHM)
            cenG = xs[np.where(ys==I0G)] # central frequency of G peak
            q = -5 # where 1/q is the Fano parameter
        
            background = min(ys)
                      
            #pbounds = ((-np.inf,1200,-np.inf,-np.inf,-np.inf,1500,-np.inf,-np.inf),
               #(np.inf,1500,np.inf,np.inf,np.inf,1700,np.inf,np.inf))
    
    
            #--------------------------find voigt fit-----------------------------   
            
            # The below is taken from the documentation on scipy.org:
            # scipy.optimize.curve_fit(f, xdata, ydata, p0)
            # uses a non-linear least squares to fit a function, f, to data.
            # f = the model function
            # xdata = the independent variable where the data is measured
            # ydata = the dependent data
            # p0 = initial guess for the parameters of f
            
            # returns: 
            #   popt: optimal values for the parameters so that the sum of the 
            #         squared residuals of f(xdata, *popt) - ydata is minimized
            #   pcov: the estimated covarience of popt. The diagonals provide the 
            #         variance of the parameter estimate.
        
            popt_LorBWF, pcov_LorBWF = scipy.optimize.curve_fit(LorBWF,
                                                                    x,
                                                                    y,
                                                                    p0=[I0D, cenD,
                                                                        gammaD, I0G,
                                                                        gammaG, cenG,
                                                                        q, background],
                                                                    method='trf')
            
            # this calculates the one standard deviation errors of the parameters
            # since var = sigma^2
            # perr_LorBWF = np.sqrt(np.diag(pcov_LorBWF))
            p0 = ['I0D','cenD','gammaD','I0G', 'gammaG', 'cenG', 'q', 'background']            
            parameters = dict()
            for j in range(len(p0)):
                parameters[p0[j]] = popt_LorBWF[j]
                
            size = int(np.sqrt(len(clean_data[i]))) # note: only valid for square maps
            fit_data = pd.DataFrame(data = np.zeros((size,size))) # initialise array
            xcoord = x/xstep
            ycoord = (y-ystep)//ystep
            fit_data[xcoord][ycoord] = parameters
            
            
            # seperate parameters for each peak
            pars_1 = popt_LorBWF[[0,1,2,-1]]
            pars_2 = popt_LorBWF[3:8]
            
            # define individual peaks based on individual parameters
            Lor_peak = Lorentzian(xs, *pars_1)
            BWF_peak = BWF(xs, *pars_2)
                          
            
            #----------------------------residuals--------------------------------
            
            # calculate the residuals (difference between fit and data points)
            residual = y - (LorBWF(xs, *popt_LorBWF))
        
        
    
#-----------------------------------------------------------------------------

# End process timer
end_time = time.process_time()
print("\nScript runtime: %.2f \bs" % (end_time - start_time))

# last runtime = 1.03s


#---------------------------------Script End----------------------------------              
