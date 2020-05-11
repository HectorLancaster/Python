#--------------------------------file notes-----------------------------------

# Creates a combined plot containing all map height histograms and their 
# normal curves. 

#--------------------------------user inputs----------------------------------

# Set the number of bins for the histograms.
nbins = 20

#------------------------------import modules---------------------------------

import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.stats as stats

#---------------------------start process timer-------------------------------

start_time = time.process_time()


#----------------------find rough location of g-peak--------------------------

max_loc = dict()
for i in material:
    size = int(np.sqrt(len(norm_data[i]))) # note: only valid for square maps
    max_loc_material = np.zeros((size,size)) # initialise array
    # -----standard-----
    xmin = int(min(raw_data[i][:,0]))
    xmax = int(max(raw_data[i][:,0]) + xstep)
    ymin = int(min(raw_data[i][:,1]))
    ymax = int(max(raw_data[i][:,1]) + ystep)
    # ------------------
    for x in range(xmin, xmax, xstep):
        for y in range(ymin, ymax, ystep):
                half = norm_data[i][xmin,ymin][:,3].shape[0]//2 # (1)
                g_slice = norm_data[i][x,y][:int(half)] # (1)
                g_loc = g_slice[g_slice.argmax(axis=0)[3]][2] # (2)
                max_loc_material[x//xstep,(y-ystep)//ystep] = g_loc # (3)
    max_loc[i] = max_loc_material # (4)
    
             
# (1) Find the length of a spectrum, half it, then use this value to slice the
#     spectra, leaving just the top half.
# (2) Find the Raman shift that corresponds to the maximum intensity, i.e. the
#     location of the g peak.
# (3) Set the g peak location to it's corresponding xy map coordinate.
# (4) Updates the parent library with all of the g peak data for each material

#----------------------------------reshape------------------------------------

# reshape the data so that it can be plotted in 2D. The orginal data format
# will be used for colour maps of g peak position at xy coordinates. 
max_loc_linear = dict()
for i in material:
    max_loc_linear[i] = np.reshape(max_loc[i], (size**2,))
       

#---------------------------------plotting------------------------------------

# initialise parameters
x0 = dict()
sigma = dict()
gaussian = dict()
histogram = dict()
counter = 0

for i in material:
    x0[i], sigma[i] = stats.norm.fit(max_loc[i]) # get mean and standev  
    # create an evenly spaced array of 100 point between max_loc range.
    xs = np.linspace(max_loc_linear[i][max_loc_linear[i].argmin()],\
                max_loc_linear[i][max_loc_linear[i].argmax()],100)
    gaussian[i] = stats.norm.pdf(xs,x0[i],sigma[i]) # create Guassian fit data
    # -----
    # differntiate between fit and histogram for each material
    counter += 1
    if counter == 1:
        line = "-."
        hcolour = "gold"
    elif counter == 2:
        line = ":"
        hcolour = "darkorange"
    elif counter == 3:
        line = "--"
        hcolour = "dodgerblue"
    # -----
    histogram[i] = plt.hist(max_loc_linear[i], bins=nbins, \
                        density=True, alpha=0.6, color=hcolour)
    plt.plot(xs,gaussian[i], line, label = i + " stats.norm.pdf", color = "k")
    plt.legend()
    plt.xlabel('Raman shift (cm⁻¹)')
    plt.ylabel('Normalised frequency')
    plt.tick_params(axis ='both', direction ='in', which = "both")
    plt.minorticks_on()

plt.savefig("C:\\Users\\Hector\\Desktop\\Data\\Figures\\height histogram.pdf")


#---------------------------------print data----------------------------------

for i in material:
    print("\n" + i + ": mean = %.0f  sd = %.0f" % (x0[i], sigma[i]))


#-----------------------------statistical tests-------------------------------

# All code in this section modified from machinelearningmaster.com
# A Gentle Introduction to Normality Tests in Python

for i in material:
    data = max_loc_linear[i]

    #-----Shapiro-Wilk Test-----
    from scipy.stats import shapiro
    stat, p = shapiro(data)
    print('\n---' + i + '---\nShapiro-Wilk Statistic: %.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
    	print('Looks Gaussian (fail to reject H0)')
    else:
    	print('Does not look Gaussian (reject H0)')
        
    #-----D'Agostino and Pearson Test----- 
    from scipy.stats import normaltest
    stat, p = normaltest(data)
    print('\nDAgostino and Pearson Statistic: %.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
    	print('Looks Gaussian (fail to reject H0)')
    else:
    	print('Does not look Gaussian (reject H0)')
        
    #-----Anderson-Darling Test-----
    from scipy.stats import anderson
    result = anderson(data)
    print('\nAnderson-Darling Statistic: %.3f' % result.statistic)
    p = 0
    for i in range(len(result.critical_values)):
    	sl, cv = result.significance_level[i], result.critical_values[i]
    	if result.statistic < result.critical_values[i]:
    		print('%.3f: %.3f, looks Gaussian (fail to reject H0))' % (sl, cv))
    	else:
    		print('%.3f: %.3f, does not look Gaussian (reject H0)' % (sl, cv))
        

        
#-----------------------------end process timer-------------------------------

# End process timer
end_time = time.process_time()
print("\nScript runtime: %.2f \bs" % (end_time - start_time))
# last runtime = 0.64s

#---------------------------------script end----------------------------------