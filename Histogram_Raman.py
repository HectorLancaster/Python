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
        array_spectra = LiC10_clean[a,b]
        array_slice = array_spectra[:int(array_spectra.shape[0]/2)]
        max_loc_val = array_slice[array_slice.argmax(axis = 0)[1]][0]
        max_loc_LiC10[a//5,(b-5)//5] = max_loc_val
        #---
        array_spectra = yp50_clean[a,b]
        array_slice = array_spectra[:int(array_spectra.shape[0]/2)]
        max_loc_val = array_slice[array_slice.argmax(axis = 0)[1]][0]
        max_loc_yp50[a//5,(b-5)//5] = max_loc_val        

#---------------------------get mean and sd info------------------------------

x0_KC10, sigma_KC10 = stats.norm.fit(max_loc_KC10)
x0_LiC10, sigma_LiC10 = stats.norm.fit(max_loc_LiC10)
x0_yp50, sigma_yp50 = stats.norm.fit(max_loc_yp50)

#----------------------------------reshape------------------------------------
        
max_loc_linear_KC10 = np.reshape(max_loc_KC10, (441,))
max_loc_linear_LiC10 = np.reshape(max_loc_LiC10, (441,))
max_loc_linear_yp50 = np.reshape(max_loc_yp50, (441,))

#------------------------define gaussian curve function-----------------------

def gauss(x,x0,sigma):
    """Generates a function to calculate the Gaussian curve"""
    y = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-((x-x0)**2)/(2*sigma**2))
    
    return y

#--------------------------------plotting-------------------------------------

nbins = 20

x = np.linspace(max_loc_linear_KC10[max_loc_linear_KC10.argmin()],\
                max_loc_linear_KC10[max_loc_linear_KC10.argmax()],100)

x0_KC10,sigma_KC10 = stats.norm.fit(max_loc_linear_KC10)
x0_LiC10,sigma_LiC10 = stats.norm.fit(max_loc_linear_LiC10)
x0_yp50,sigma_yp50 = stats.norm.fit(max_loc_linear_yp50)

gaussian_yp50 = stats.norm.pdf(x,x0_yp50,sigma_yp50)
histogram_yp50 = plt.hist(max_loc_linear_yp50, bins = nbins, density=True,alpha=0.6, color = "r")
gaussian_KC10 = stats.norm.pdf(x,x0_KC10,sigma_KC10)
histogram_KC10 = plt.hist(max_loc_linear_KC10, bins = nbins, density=True,alpha=0.4, color = "b")
gaussian_LiC10 = stats.norm.pdf(x,x0_LiC10,sigma_LiC10)
histogram_LiC10 = plt.hist(max_loc_linear_LiC10, bins = nbins, density=True,alpha=0.4, color = "g")


plt.plot(x,gaussian_KC10,'b-.', label="KC10 stats.norm.pdf")
plt.plot(x,gaussian_LiC10,'g-.', label="LiC10 stats.norm.pdf")
plt.plot(x,gaussian_yp50,'r-.', label="yp50 stats.norm.pdf")
plt.legend()
plt.xlabel('Raman shift (cm⁻¹)')
plt.ylabel('normalised frequency')
#title_label= (r'yp50: Line fitted with Gaussian $x_0$ =' \
              #'{0:.4n}, $\sigma$ = {1:.1n}'.format(x0_yp50,sigma_yp50))

#plt.title(title_label)
plt.savefig("C:\\Users\\Hector\\Desktop\\Data\\height histogram.pdf")

#-----------------------------------------------------------------------------

print("\nyp50: mean = %.0f  sd = %.0f" % (x0_yp50, sigma_yp50))
print("KC10: mean = %.0f  sd = %.0f" % (x0_KC10, sigma_KC10))
print("LiC10: mean = %.0f  sd = %.0f" % (x0_LiC10, sigma_LiC10))

#-----------------------------statistical tests-------------------------------
# All code in this section taken from machinelearningmaster.com
# A Gentle Introduction to Normality Tests in Python

#----------------yp50----------------
data = max_loc_linear_yp50

# Shapiro-Wilk Test.
from scipy.stats import shapiro
stat, p = shapiro(data)
print('\n---yp50---\nShapiro-Wilk Statistics=%.3f, p=%.3f' % (stat, p))
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
        
#----------------KC10----------------
data = max_loc_linear_KC10
# Shapiro-Wilk Test.
from scipy.stats import shapiro
stat, p = shapiro(data)
print('\n---KC10---\nShapiro-Wilk Statistics=%.3f, p=%.3f' % (stat, p))
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

#----------------LiC10----------------
data = max_loc_linear_LiC10

# Shapiro-Wilk Test.
from scipy.stats import shapiro
stat, p = shapiro(data)
print('\n---LiC10---\nShapiro-Wilk Statistics=%.3f, p=%.3f' % (stat, p))
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
# last runtime = 0.2s
