#--------------------------------File notes-----------------------------------

# no way here of removing outliers, very important this is done otherwise
# the x coord coresponding to the max y could represent the x coord of the 
# outlier.

#------------------------------import modules---------------------------------

import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.stats as stats

#---------------------------start process timer-------------------------------

start_time = time.process_time()

#----------------------find rough location of g-peak--------------------------

max_loc_KC10 = np.zeros((21,21))
max_loc_LiC10 = np.zeros((21,21))
max_loc_yp50 = np.zeros((21,21))
for a in range(0,105,5):
    for b in range(-100,5,5):
        array_spectra = KC10_clean[a,b]
        array_slice = array_spectra[:int(array_spectra.shape[0]/2)]
        max_loc_val = array_slice[array_slice.argmax(axis = 0)[1]][0]
        max_loc_KC10[a//5,(b-5)//5] = max_loc_val
        #---
        #array_spectra = np.array(spectra_LiC10[a,b])
        #array_slice = array_spectra[:int(array_spectra.shape[0]/2)]
        #max_loc_val = array_slice[array_slice.argmax(axis = 0)[1]][0]
        #max_loc_LiC10[a//5,(b-5)//5] = max_loc_val
        #---
        #array_spectra = np.array(spectra_yp50[a,b])
        #array_slice = array_spectra[:int(array_spectra.shape[0]/2)]
        #max_loc_val = array_slice[array_slice.argmax(axis = 0)[1]][0]
        #max_loc_yp50[a//5,(b-5)//5] = max_loc_val        

#---------------------------get mean and sd info------------------------------

x0, sigma = stats.norm.fit(max_loc)

#-------------------------limit range of histogram----------------------------

max_loc = np.zeros((21,21))
for a in range(0,105,5):
    for b in range(-100,5,5):
        array_spectra = np.array(spectra_KC10[a,b])
        temp = list()
        for i in range(576):       
            if array_spectra[i,0] > (x0 - 3*sigma) \
                and array_spectra[i,0] < (x0 + 3*sigma):
                    temp.append([array_spectra[i,0],array_spectra[i,1]])
        array_slice = np.array(temp)
        max_loc_val = array_slice[array_slice.argmax(axis = 0)[1]][0]
        max_loc[a//5,(b-5)//5] = max_loc_val
        
max_loc_linear = np.reshape(max_loc, (441,))

#------------------------define gaussian curve function-----------------------

def gauss(x,x0,sigma):
    """Generates a function to calculate the Gaussian curve"""
    y = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-((x-x0)**2)/(2*sigma**2))
    
    return y

#--------------------------------plotting-------------------------------------

x = np.linspace(max_loc_linear[max_loc_linear.argmin()],\
                max_loc_linear[max_loc_linear.argmax()],100)
x0,sigma = stats.norm.fit(max_loc_linear)

gaussian = stats.norm.pdf(x,x0,sigma)
histogram = plt.hist(max_loc_linear, bins = 20, density=True,alpha=0.25)
plt.plot(x,gaussian,'k-.', label="stats.norm.pdf")
plt.legend()
plt.xlabel('Raman shift (cm⁻¹)')
plt.ylabel('normalised frequency')
title_label= (r'KC10: Line fitted with Gaussian $x_0$ =' \
              '{0:.4n}, $\sigma$ = {1:.1n}'.format(x0,sigma))
plt.title(title_label)
plt.savefig("C:\\Users\\Hector\\Desktop\\Data\\height histogram.pdf")

#-----------------------------statistical tests-------------------------------

# All code in this section taken from machinelearningmaster.com
# A Gentle Introduction to Normality Tests in Python

data = max_loc_linear

# Shapiro-Wilk Test.
from scipy.stats import shapiro
stat, p = shapiro(data)
print('\nShapiro-Wilk Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Sample looks Gaussian (fail to reject H0)')
else:
	print('Sample does not look Gaussian (reject H0)')
    
# DAgostino and Pearson Test.        
from scipy.stats import normaltest
stat, p = normaltest(data)
print('\nDAgostino and Pearson Statistic Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Sample looks Gaussian (fail to reject H0)')
else:
	print('Sample does not look Gaussian (reject H0)')
    
# Anderson-Darling Test.
from scipy.stats import anderson
result = anderson(data)
print('\nAnderson-Darling Statistic: %.3f' % result.statistic)
p = 0
for i in range(len(result.critical_values)):
	sl, cv = result.significance_level[i], result.critical_values[i]
	if result.statistic < result.critical_values[i]:
		print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
	else:
		print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
        
#-----------------------------end process timer-------------------------------

end_time = time.process_time()
print("\nScript runtime:", str(end_time - start_time), "\bs")
# last runtime = 0.6s
