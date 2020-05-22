#--------------------------------file notes-----------------------------------

# Applies a voigt fit to two peak data

# To do:
#       1) try to fit 3 peaks and see what happens
#       2) perhaps convert the code to n-peaks


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
    
    
def _1Voigt(x, ampG1, cen1, sigmaG1, ampL1, widL1):
    return (ampG1*(1/(sigmaG1*(np.sqrt(2*np.pi))))*(np.exp(-((x-cen1)**2)/(2*sigmaG1**2)))) +\
            ((ampL1*widL1**2/((x-cen1)**2+widL1**2)) )
    
                  
def _2Voigt(x, ampG1, cenG1, sigmaG1, ampL1, cenL1, widL1,
            ampG2, cenG2, sigmaG2, ampL2, cenL2, widL2):
    return (ampG1*(1/(sigmaG1*(np.sqrt(2*np.pi))))*(np.exp(-((x-cenG1)**2)/(2*sigmaG1**2)))) +\
            ((ampL1*widL1**2/((x-cenL1)**2+widL1**2)) ) +\
                (ampG2*(1/(sigmaG2*(np.sqrt(2*np.pi))))*(np.exp(-((x-cenG2)**2)/(2*sigmaG2**2)))) +\
                    ((ampL2*widL2**2/((x-cenL2)**2+widL2**2)) )

#--------------------------get & print fitting data---------------------------

for i in material:
    x = norm_data[i][0,-100][:,2]
    y = av_data[i]
    baseline = min(y)
    y = y - baseline
    
    #-----------------------------initial guesses-----------------------------    
    ampG1 = max(y)
    ampL1 = ampG1 # height of peak
    cen1 = x[y.argmax(axis=0)]
    #cenL1 = cenG1
    sigmaG1 = 40
    widL1 = sigmaG1
    
    ampG2 = max(y[:288])
    ampL2 = ampG2
    cen2 = x[y[:288].argmax(axis=0)]
    #cenL2 = cenG2
    sigmaG2 = 30
    widL2 = sigmaG2
                
    #-------------------------------find voigt fit----------------------------   
    popt_2voigt, pcov_2voigt = scipy.optimize.curve_fit(_2Voigt,
                                                            x,
                                                            y,
                                                            p0=[ampG1, cenG1, sigmaG1,
                                                                ampL1, cenL1, widL1,
                                                                ampG2, cenG2, sigmaG2,
                                                                ampL2, cenL2, widL2],
                                                            bounds = (0,np.inf))
    perr_2voigt = np.sqrt(np.diag(pcov_2voigt))
    
    pars_1 = popt_2voigt[6:12]
    pars_2 = popt_2voigt[0:6]
    
    voigt_peak_1 = _1Voigt(x, *pars_1)
    voigt_peak_2 = _1Voigt(x, *pars_2)
                  
    
    #----------------------------------print----------------------------------
    
    # note: need to double check that the peak1/2 data is indexed right for pars &
    #       perr
    
    # this cell prints the fitting parameters with their errors
    print("\n--" + i + "--")
    print("-------------Peak 1-------------")
    print("amplitude = %0.2f (+/-) %0.2f" % (pars_1[0], perr_2voigt[3]))
    print("center = %0.2f (+/-) %0.2f" % (pars_1[1], perr_2voigt[4]))
    print("width = %0.2f (+/-) %0.2f" % (pars_1[2], perr_2voigt[5]))
    print("area = %0.2f" % np.trapz(voigt_peak_1))
    print("--------------------------------")
    print("-------------Peak 2-------------")
    print("amplitude = %0.2f (+/-) %0.2f" % (pars_2[0], perr_2voigt[0]))
    print("center = %0.2f (+/-) %0.2f" % (pars_2[1], perr_2voigt[1]))
    print("width = %0.2f (+/-) %0.2f" % (pars_2[2], perr_2voigt[2]))
    print("area = %0.2f" % np.trapz(voigt_peak_2))
    
    
    #---------------------------------residuals-------------------------------
    
    residual_2 = y - (_2Voigt(x, *popt_2voigt))
    
    
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
    ax1.plot(x, voigt_peak_1, color="orange", alpha=0.5)
    ax1.fill_between(x, voigt_peak_1.min(), voigt_peak_1,
                     facecolor="orange", alpha=0.5)
      
    # peak 2
    ax1.plot(x, voigt_peak_2, color="cornflowerblue", alpha=0.5)
    ax1.fill_between(x, voigt_peak_2.min(), voigt_peak_2,
                     facecolor="cornflowerblue", alpha=0.5)  
    
    # fit: scipy.optimize.curve_fit
    ax1.plot(x, _2Voigt(x, *popt_2voigt), 'k--', label= "curvefit")
    
    # residuals
    ax2.plot(x, residual_2, "o", markersize = 1, color = "gray")              
    
    # plot specifications
    ax2.xaxis.set_minor_locator(AutoMinorLocator(4)) # number of minor ticks
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(100))
    #ax1.xaxis.set_major_formatter(plt.NullFormatter())
    
    ax2.set_xlabel('Raman shift (cm⁻¹)')
    ax1.set_ylabel('Intensity (arb. units)', labelpad=11)
    ax2.set_ylabel('Res.', labelpad = 0.2)
    ax1.legend(loc="best", fontsize=8)
    ax1.set_yticks([],[])
    
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

# last runtime = 4.25s


#---------------------------------Script End----------------------------------              
