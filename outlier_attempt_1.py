# -*- coding: utf-8 -*-
"""
Created on Mon May  4 10:07:49 2020

@author: Hector
"""
import statistics
import numpy as np

n_columns = len(spectra_KC10)
n_rows =  len(spectra_KC10[0,-100])
outliers_KC10 = np.zeros((n_rows,n_columns))
outliers_LiC10 = np.zeros((n_rows,n_columns))
outliers_yp50 = np.zeros((n_rows,n_columns))

counter = 0
for a in range(0,105,5):            
    for b in range(-100,5,5):
        for i in range(n_rows):
            outliers_KC10[i][counter] = spectra_KC10[a,b][i][1]
            outliers_LiC10[i][counter] = spectra_LiC10[a,b][i][1]
            outliers_yp50[i][counter] = spectra_yp50[a,b][i][1]
        counter += 1
 
KC10_outliers_av  = np.zeros(n_rows,)        
KC10_outliers_sd = np.zeros(n_rows,)
LiC10_outliers_av = np.zeros(n_rows,)  
LiC10_outliers_sd = np.zeros(n_rows,)  
yp50_outliers_av = np.zeros(n_rows,)  
yp50_outliers_sd = np.zeros(n_rows,)  
for i in range(n_rows):
    KC10_outliers_av[i] = statistics.mean(outliers_KC10[i])
    KC10_outliers_sd[i] = statistics.stdev(outliers_KC10[i])
    LiC10_outliers_av[i] = statistics.mean(outliers_LiC10[i])
    LiC10_outliers_sd[i] = statistics.stdev(outliers_LiC10[i])    
    yp50_outliers_av[i] = statistics.mean(outliers_yp50[i])
    yp50_outliers_sd[i] = statistics.stdev(outliers_yp50[i])
    
for j in range(n_columns):
    for i in range(n_rows):
        if KC10_outliers_av[i] + 3 * KC10_outliers_sd[i] < outliers_KC10[i][j]\
            < KC10_outliers_av[i] - 3 * KC10_outliers_sd[i]:
            outliers_KC10[i][j] = np.nan
            
for j in range(n_columns):
    for i in range(n_rows):
        if LiC10_outliers_av[i] + 3 * LiC10_outliers_sd[i] < outliers_LiC10[i][j]\
            < LiC10_outliers_av[i] - 3 * LiC10_outliers_sd[i]:
            outliers_LiC10[i][j] = np.nan
            
for j in range(n_columns):
    for i in range(n_rows):
        if yp50_outliers_av[i] + 3 * yp50_outliers_sd[i] < outliers_yp50[i][j]\
            < yp50_outliers_av[i] - 3 * yp50_outliers_sd[i]:
            outliers_yp50[i][j] = np.nan

x_coord = np.zeros(n_rows,)
for i in range(n_rows):
    x_coord[i] = spectra_KC10[10,-100][i][0]

counter = 0
for a in range(0,105,5):            
    for b in range(-100,5,5):
            spectra_KC10[a,b] = [[x_coord[i], outliers_KC10[i][counter]] for i in range(n_rows)]
            spectra_LiC10[a,b] = [[x_coord[i], outliers_LiC10[i][counter]] for i in range(n_rows)]
            spectra_yp50[a,b] = [[x_coord[i], outliers_yp50[i][counter]] for i in range(n_rows)]
            counter += 1
            
